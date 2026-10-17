"""Synchronous and asynchronous API clients on top of HTTPX.

Design points:

- One ``httpx.Client`` (connection pool) per ``ApiClient``; close it with
  ``close()`` or a ``with`` block. Nothing connects at construction time.
- HTTP error statuses are returned, never raised: a 404 or 500 is a response
  to assert on. Only transport failures raise (``ApiTransportError``).
- Redirects are followed by this module, not by HTTPX, so credentials
  (``Authorization``, cookies, API-key headers) are dropped when a redirect
  leaves the original origin. Following is off unless enabled.
- Retries are off unless a ``RetryConfig`` is given. They apply only to
  idempotent methods, or to requests with an idempotency key, or when the
  caller passes ``idempotent=True``.
- ``trust_env`` is disabled: proxies and ``.netrc`` credentials are never
  picked up implicitly; configure them explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import ssl
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from complete_automation.api.auth import build_auth
from complete_automation.api.diagnostics import Sanitizer
from complete_automation.api.errors import (
    ApiTimeoutError,
    ApiTransportError,
    RedirectLoopError,
    ResponseDecodeError,
    UnsafeOperationError,
)
from complete_automation.api.settings import ApiRetrySettings, ApiSettings
from complete_automation.core.context import current_test_context, new_correlation_id
from complete_automation.core.exceptions import ConfigurationError
from complete_automation.core.log import get_logger
from complete_automation.core.redaction import Redactor
from complete_automation.core.retry import RetryPolicy

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_NEVER_RETRIED_BY_METHOD = frozenset({"POST", "PATCH"})
_RETRYABLE_4XX = frozenset({408, 409, 425, 429})
_PATH_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_TRANSPORT_ERRORS: tuple[type[httpx.TransportError], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


UNSET: Any = _Unset()
_log = get_logger("api.client")


# --------------------------------------------------------------------------- retry


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (moment - (now or datetime.now(UTC))).total_seconds())


@dataclass(frozen=True)
class RetryConfig:
    """When and how often a request is repeated.

    Status codes must be 5xx or one of 408, 409, 425, 429: authentication and
    validation failures (401, 403, 422, ...) are not retryable. POST and PATCH
    cannot be listed in ``methods``; repeat them only with an idempotency key or
    ``idempotent=True`` on the request.
    """

    max_attempts: int = 3
    delay_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 10.0
    jitter_ratio: float = 0.0
    status_codes: frozenset[int] = frozenset({429, 502, 503, 504})
    methods: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})
    transport_errors: tuple[type[httpx.TransportError], ...] = DEFAULT_TRANSPORT_ERRORS
    respect_retry_after: bool = True
    max_retry_after_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_codes", frozenset(self.status_codes))
        object.__setattr__(self, "methods", frozenset(m.upper() for m in self.methods))
        for code in self.status_codes:
            if not 400 <= code <= 599 or (code < 500 and code not in _RETRYABLE_4XX):
                raise ValueError(f"status {code} is not retryable (allowed: 5xx, {sorted(_RETRYABLE_4XX)})")
        blocked = self.methods & _NEVER_RETRIED_BY_METHOD
        if blocked:
            raise ValueError(
                f"{sorted(blocked)} cannot be retried by method; use an idempotency key or idempotent=True"
            )
        for kind in self.transport_errors:
            if not (isinstance(kind, type) and issubclass(kind, httpx.TransportError)):
                raise TypeError(f"transport_errors must be httpx.TransportError types, got {kind!r}")
        if self.max_retry_after_seconds < 0:
            raise ValueError("max_retry_after_seconds must not be negative")
        self._policy()  # validates attempts, delays, backoff and jitter

    def _policy(self) -> RetryPolicy:
        return RetryPolicy(
            retry_on=(httpx.TransportError,),
            max_attempts=self.max_attempts,
            delay_seconds=self.delay_seconds,
            backoff_multiplier=self.backoff_multiplier,
            max_delay_seconds=self.max_delay_seconds,
            jitter_ratio=self.jitter_ratio,
        )

    def delay_before(self, attempt: int, response: httpx.Response | None, rng: random.Random | None) -> float:
        if response is not None and self.respect_retry_after:
            wait = parse_retry_after(response.headers.get("Retry-After"))
            if wait is not None:
                return min(wait, self.max_retry_after_seconds)
        return self._policy().delay_before(attempt, rng)

    @classmethod
    def from_settings(cls, settings: ApiRetrySettings) -> RetryConfig | None:
        if not settings.enabled:
            return None
        return cls(
            max_attempts=settings.max_attempts,
            delay_seconds=settings.delay_seconds,
            backoff_multiplier=settings.backoff_multiplier,
            max_delay_seconds=settings.max_delay_seconds,
            jitter_ratio=settings.jitter_ratio,
            status_codes=frozenset(settings.status_codes),
            methods=frozenset(settings.methods),
            respect_retry_after=settings.respect_retry_after,
            max_retry_after_seconds=settings.max_retry_after_seconds,
        )


@dataclass(frozen=True)
class AttemptRecord:
    number: int
    status_code: int | None
    error: str | None
    duration_ms: float
    """Client-observed time of this attempt, including followed redirects."""
    wait_ms: float = 0.0
    """Back-off waited after this attempt before the next one."""


# --------------------------------------------------------------------------- response


class ApiResponse:
    """An HTTP response plus what the client knows about how it was obtained.

    ``raw`` is the underlying ``httpx.Response`` for anything not covered here.
    The body is never decoded implicitly; call ``json()`` or ``text`` when the
    content type is appropriate.
    """

    def __init__(
        self,
        raw: httpx.Response,
        *,
        method: str,
        operation: str,
        attempts: tuple[AttemptRecord, ...],
        redirects: tuple[str, ...],
        total_ms: float,
        sanitizer: Sanitizer,
    ) -> None:
        self.raw = raw
        self.method = method
        self.operation = operation
        """The path template as requested, e.g. ``/users/{user_id}`` (no IDs or query)."""
        self.attempts = attempts
        self.redirects = redirects
        self.total_ms = total_ms
        """Wall time of the whole call: every attempt plus back-off waits."""
        self.sanitizer = sanitizer

    @property
    def status_code(self) -> int:
        return self.raw.status_code

    @property
    def headers(self) -> httpx.Headers:
        return self.raw.headers

    @property
    def cookies(self) -> httpx.Cookies:
        return self.raw.cookies

    @property
    def content(self) -> bytes:
        return self.raw.content

    @property
    def text(self) -> str:
        return self.raw.text

    @property
    def url(self) -> str:
        """Final URL, sanitized for display."""
        return self.sanitizer.url(self.raw.url)

    @property
    def latency_ms(self) -> float:
        """Client-observed latency of the final attempt (network plus server time)."""
        return self.attempts[-1].duration_ms

    @property
    def retries(self) -> int:
        return len(self.attempts) - 1

    @property
    def is_redirect(self) -> bool:
        return self.raw.is_redirect

    @property
    def content_type(self) -> str | None:
        value = self.raw.headers.get("Content-Type")
        return value.split(";")[0].strip().lower() if value else None

    @property
    def is_json(self) -> bool:
        kind = self.content_type or ""
        return kind == "application/json" or kind.endswith("+json")

    def json(self) -> Any:
        """Decoded JSON body; raises ``ResponseDecodeError`` (an AssertionError) if impossible."""
        if not self.raw.content:
            raise ResponseDecodeError(f"{self.describe()}: expected a JSON body, got an empty body")
        try:
            return self.raw.json()
        except ValueError as error:
            raise ResponseDecodeError(
                f"{self.describe()}: body is not valid JSON ({error}); "
                f"content type {self.content_type or 'missing'}; body: {self.body_excerpt(200)}"
            ) from None

    def server_timing(self) -> dict[str, float]:
        """Server-reported durations from the ``Server-Timing`` header, in milliseconds."""
        timings: dict[str, float] = {}
        for entry in self.raw.headers.get_list("Server-Timing", split_commas=True):
            name, *params = [part.strip() for part in entry.split(";")]
            for param in params:
                key, _, value = param.partition("=")
                if key.strip().lower() == "dur":
                    try:
                        timings[name] = float(value)
                    except ValueError:
                        pass
        return timings

    def body_excerpt(self, limit: int | None = 500) -> str:
        return self.sanitizer.body(self.raw.content, self.raw.headers.get("Content-Type"), limit)

    def describe(self) -> str:
        retries = f", {self.retries} retr{'y' if self.retries == 1 else 'ies'}" if self.retries else ""
        return f"{self.method} {self.operation} -> {self.status_code} ({self.latency_ms:.1f} ms{retries})"

    def to_record(self, *, include_bodies: bool = True, body_limit: int = 65536) -> dict[str, Any]:
        """A sanitized, JSON-compatible description of the exchange (for artifacts)."""
        request = self.raw.request
        record: dict[str, Any] = {
            "operation": self.operation,
            "method": self.method,
            "request": {
                "url": self.sanitizer.url(request.url),
                "headers": self.sanitizer.headers(request.headers),
            },
            "response": {
                "status_code": self.status_code,
                "url": self.url,
                "headers": self.sanitizer.headers(self.raw.headers),
                "content_type": self.content_type,
            },
            "latency_ms": round(self.latency_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "attempts": [
                {
                    "number": a.number,
                    "status_code": a.status_code,
                    "error": a.error,
                    "duration_ms": round(a.duration_ms, 3),
                    "wait_ms": round(a.wait_ms, 3),
                }
                for a in self.attempts
            ],
            "redirects": list(self.redirects),
        }
        if include_bodies:
            try:
                request_body = self.sanitizer.body(
                    request.content, request.headers.get("Content-Type"), body_limit
                )
            except httpx.RequestNotRead:
                request_body = "<streamed body not recorded>"
            record["request"]["body"] = request_body
            record["response"]["body"] = self.body_excerpt(body_limit)
        return record

    def __repr__(self) -> str:
        return f"<ApiResponse {self.describe()}>"


# --------------------------------------------------------------------------- shared client logic


@dataclass
class _Plan:
    method: str
    url: str
    operation: str
    headers: dict[str, str]
    params: Any
    body: dict[str, Any]
    timeout: Any
    auth: Any
    follow: bool
    retry: RetryConfig | None
    repeatable: bool


class _ClientBase:
    def __init__(
        self,
        base_url: str,
        *,
        auth: httpx.Auth | None = None,
        timeout: float | httpx.Timeout = httpx.Timeout(30.0, connect=5.0, pool=5.0),
        verify: bool | ssl.SSLContext = True,
        headers: Mapping[str, str] | None = None,
        cookies: Mapping[str, str] | None = None,
        follow_redirects: bool = False,
        max_redirects: int = 5,
        proxy: str | None = None,
        retry: RetryConfig | None = None,
        read_only: bool = False,
        correlation_header: str | None = "X-Correlation-ID",
        redactor: Redactor | None = None,
        sensitive_names: tuple[str, ...] = (),
        log_bodies: bool = False,
        on_response: Callable[[ApiResponse], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not base_url.lower().startswith(("http://", "https://")):
            raise ConfigurationError(f"base_url must be an absolute http(s) URL, got {base_url!r}")
        if timeout is None:
            raise ConfigurationError("timeouts must be finite")
        if not 0 <= max_redirects <= 20:
            raise ConfigurationError("max_redirects must be between 0 and 20")
        self.base_url = httpx.URL(base_url)
        self.follow_redirects = follow_redirects
        self.max_redirects = max_redirects
        self.retry = retry
        self.read_only = read_only
        self.correlation_header = correlation_header
        self.log_bodies = log_bodies
        self.on_response = on_response
        self._rng = rng
        names = tuple(sensitive_names) + tuple(getattr(auth, "sensitive_names", ()))
        self.sanitizer = Sanitizer(redactor, names)
        if verify is False:
            _log.warning("TLS certificate verification is disabled for %s", self.sanitizer.url(base_url))
        self._options: dict[str, Any] = dict(
            base_url=self.base_url,
            auth=auth,
            timeout=timeout,
            verify=verify,
            headers=dict(headers or {}),
            cookies=dict(cookies or {}),
            follow_redirects=False,
            proxy=proxy,
            trust_env=False,
        )

    @classmethod
    def from_settings(
        cls,
        settings: ApiSettings,
        *,
        environment: str,
        redactor: Redactor | None = None,
        base_url: str | None = None,
        auth: Any = UNSET,
        **overrides: Any,
    ) -> Any:
        """Build a client from ``modules.api`` settings; keyword overrides win."""
        url = base_url or settings.base_url
        if not url:
            raise ConfigurationError("modules.api.base_url is not set")
        verify: bool | ssl.SSLContext = settings.verify_tls
        if settings.verify_tls and settings.ca_bundle is not None:
            verify = ssl.create_default_context(cafile=str(settings.ca_bundle))
        options: dict[str, Any] = dict(
            auth=build_auth(settings.auth) if auth is UNSET else auth,
            timeout=httpx.Timeout(
                connect=settings.timeouts.connect,
                read=settings.timeouts.read,
                write=settings.timeouts.write,
                pool=settings.timeouts.pool,
            ),
            verify=verify,
            headers=settings.default_headers,
            follow_redirects=settings.follow_redirects,
            max_redirects=settings.max_redirects,
            proxy=settings.proxy.get_secret_value() if settings.proxy else None,
            retry=RetryConfig.from_settings(settings.retry),
            read_only=settings.is_read_only(environment),
            correlation_header=settings.correlation_header,
            redactor=redactor,
            log_bodies=settings.log_bodies,
        )
        options.update(overrides)
        return cls(url, **options)

    # -- request planning ----------------------------------------------------------------

    def _plan(
        self,
        method: str,
        path: str,
        *,
        path_params: Mapping[str, Any] | None,
        params: Any,
        headers: Mapping[str, str] | None,
        cookies: Mapping[str, str] | None,
        json: Any,
        data: Any,
        files: Any,
        content: Any,
        timeout: Any,
        auth: Any,
        follow_redirects: bool | None,
        retry: Any,
        idempotency_key: str | None,
        idempotent: bool,
    ) -> _Plan:
        method = method.upper()
        operation = path
        url = _render_path(path, path_params)
        if url.lower().startswith(("http://", "https://")) and not _same_origin(httpx.URL(url), self.base_url):
            raise UnsafeOperationError(
                f"{method} {self.sanitizer.url(url)} is outside {self.sanitizer.url(self.base_url)}; "
                "create a separate client for another origin"
            )
        if self.read_only and method in MUTATING_METHODS:
            raise UnsafeOperationError(
                f"{method} {operation} refused: this client is read-only "
                "(protected environment, or modules.api.read_only is true)"
            )
        merged = dict(headers or {})
        if self.correlation_header and not _has_header(merged, self.correlation_header):
            context = current_test_context()
            merged[self.correlation_header] = context.correlation_id if context else new_correlation_id()
        if idempotency_key is not None:
            merged["Idempotency-Key"] = idempotency_key
        if cookies:
            existing = [value for key, value in merged.items() if key.lower() == "cookie"]
            merged = {key: value for key, value in merged.items() if key.lower() != "cookie"}
            merged["Cookie"] = "; ".join(existing + [f"{key}={value}" for key, value in cookies.items()])
        body: dict[str, Any] = {}
        if json is not UNSET:
            body["json"] = json
        for key, value in (("data", data), ("files", files), ("content", content)):
            if value is not None:
                body[key] = value
        resolved_retry = self.retry if retry is UNSET else (retry or None)
        repeatable = bool(
            resolved_retry
            and (method in resolved_retry.methods or idempotency_key is not None or idempotent)
        )
        return _Plan(
            method=method,
            url=url,
            operation=operation,
            headers=merged,
            params=params,
            body=body,
            timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout,
            auth=httpx.USE_CLIENT_DEFAULT if auth is UNSET else auth,
            follow=self.follow_redirects if follow_redirects is None else follow_redirects,
            retry=resolved_retry,
            repeatable=repeatable,
        )

    def _retry_status(self, plan: _Plan, response: httpx.Response, attempt: int) -> bool:
        return bool(
            plan.repeatable
            and plan.retry
            and attempt < plan.retry.max_attempts
            and response.status_code in plan.retry.status_codes
        )

    def _retry_error(self, plan: _Plan, error: httpx.TransportError, attempt: int) -> bool:
        return bool(
            plan.repeatable
            and plan.retry
            and attempt < plan.retry.max_attempts
            and isinstance(error, plan.retry.transport_errors)
        )

    def _redirect(self, response: httpx.Response, hops: list[str], plan: _Plan) -> tuple[httpx.Request, Any] | None:
        """The next hop of a redirect chain, or None to stop; credentials stay on their origin."""
        if not (plan.follow and response.is_redirect and response.next_request is not None):
            return None
        if len(hops) >= self.max_redirects:
            raise RedirectLoopError(
                f"{plan.method} {plan.operation}: more than {self.max_redirects} redirects "
                f"(last: {self.sanitizer.url(response.request.url)})"
            )
        hops.append(f"{response.status_code} {self.sanitizer.url(response.request.url)}")
        nxt = response.next_request
        if _same_origin(nxt.url, self.base_url):
            return nxt, plan.auth
        for name in list(nxt.headers):
            if self.sanitizer.is_sensitive(name):
                del nxt.headers[name]
        return nxt, None  # no authentication on another origin

    def _transport_error(self, plan: _Plan, error: httpx.TransportError, attempts: int) -> ApiTransportError:
        kind = ApiTimeoutError if isinstance(error, httpx.TimeoutException) else ApiTransportError
        url = self.sanitizer.url(self._display_url(plan.url))
        detail = self.sanitizer.redactor.redact_text(str(error)) or type(error).__name__
        return kind(
            f"{plan.method} {plan.operation} ({url}): {type(error).__name__}: {detail} "
            f"after {attempts} attempt(s)",
            method=plan.method,
            url=url,
            attempts=attempts,
        )

    def _display_url(self, url: str) -> str:
        """The request URL as HTTPX builds it: relative paths extend the base URL's path."""
        if url.lower().startswith(("http://", "https://")):
            return url
        return f"{str(self.base_url).rstrip('/')}/{url.lstrip('/')}"

    def _finish(
        self,
        plan: _Plan,
        raw: httpx.Response,
        attempts: list[AttemptRecord],
        hops: list[str],
        started: float,
    ) -> ApiResponse:
        response = ApiResponse(
            raw,
            method=plan.method,
            operation=plan.operation,
            attempts=tuple(attempts),
            redirects=tuple(hops),
            total_ms=(time.perf_counter() - started) * 1000,
            sanitizer=self.sanitizer,
        )
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("%s [%s]", response.describe(), response.url)
            if self.log_bodies:
                _log.debug("response body: %s", response.body_excerpt())
        if self.on_response is not None:
            self.on_response(response)
        return response


def _render_path(path: str, values: Mapping[str, Any] | None) -> str:
    names = _PATH_PARAM.findall(path)
    values = dict(values or {})
    missing = [name for name in names if name not in values]
    unexpected = sorted(set(values) - set(names))
    if missing or unexpected:
        raise ValueError(f"path {path!r}: missing parameter(s) {missing}, unexpected parameter(s) {unexpected}")
    for name in names:
        if values[name] is None or values[name] == "":
            raise ValueError(f"path {path!r}: parameter {name!r} is empty")
    return _PATH_PARAM.sub(lambda match: quote(str(values[match.group(1)]), safe=""), path)


def path_template_regex(template: str) -> re.Pattern[str]:
    """Regex for a path template: literal parts escaped, each ``{name}`` matches one segment."""
    parts, last = [], 0
    for match in _PATH_PARAM.finditer(template):
        parts += [re.escape(template[last:match.start()]), f"(?P<{match.group(1)}>[^/]+)"]
        last = match.end()
    parts.append(re.escape(template[last:]))
    return re.compile("^" + "".join(parts) + "$")


def _same_origin(url: httpx.URL, other: httpx.URL) -> bool:
    return (url.scheme, url.host, url.port) == (other.scheme, other.host, other.port)


def _has_header(headers: Mapping[str, str], name: str) -> bool:
    return any(key.lower() == name.lower() for key in headers)


_REQUEST_DOC = """Send ``method path`` and return the response; HTTP errors are not raised.

    Args:
        path: path relative to ``base_url`` (``/users/{user_id}``) or an absolute
            URL on the same origin.
        path_params: values for ``{name}`` placeholders, URL-encoded.
        params, headers, cookies: query parameters, extra headers, cookies.
        json, data, files, content: body (JSON, form fields, multipart files, raw).
        timeout: seconds or ``httpx.Timeout`` for this request.
        auth: ``None`` sends this request without authentication.
        follow_redirects: override the client setting.
        retry: a ``RetryConfig``, or ``None``/``False`` to disable retries.
        idempotency_key: sent as ``Idempotency-Key``; makes POST/PATCH retryable.
        idempotent: the caller asserts this request is safe to repeat.
"""


# --------------------------------------------------------------------------- sync client


class ApiClient(_ClientBase):
    """Synchronous API client. Use as a context manager or call ``close()``."""

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, **options: Any) -> None:
        super().__init__(base_url, **options)
        self._sleep = sleep
        self._client = httpx.Client(transport=transport, **self._options)

    @property
    def httpx_client(self) -> httpx.Client:
        """The underlying client, for features this wrapper does not cover."""
        return self._client

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def request(self, method: str, path: str, *, path_params: Mapping[str, Any] | None = None,
                params: Any = None, headers: Mapping[str, str] | None = None,
                cookies: Mapping[str, str] | None = None, json: Any = UNSET, data: Any = None,
                files: Any = None, content: Any = None, timeout: Any = None, auth: Any = UNSET,
                follow_redirects: bool | None = None, retry: Any = UNSET,
                idempotency_key: str | None = None, idempotent: bool = False) -> ApiResponse:
        plan = self._plan(method, path, path_params=path_params, params=params, headers=headers,
                          cookies=cookies, json=json, data=data, files=files, content=content,
                          timeout=timeout, auth=auth, follow_redirects=follow_redirects, retry=retry,
                          idempotency_key=idempotency_key, idempotent=idempotent)
        attempts: list[AttemptRecord] = []
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            hops: list[str] = []
            began = time.perf_counter()
            try:
                raw = self._send(plan, hops)
            except httpx.TransportError as error:
                elapsed = (time.perf_counter() - began) * 1000
                if not self._retry_error(plan, error, attempt):
                    attempts.append(AttemptRecord(attempt, None, type(error).__name__, elapsed))
                    raise self._transport_error(plan, error, attempt) from error
                wait = plan.retry.delay_before(attempt + 1, None, self._rng)  # type: ignore[union-attr]
                attempts.append(AttemptRecord(attempt, None, type(error).__name__, elapsed, wait * 1000))
                self._sleep(wait)
                continue
            elapsed = (time.perf_counter() - began) * 1000
            if self._retry_status(plan, raw, attempt):
                wait = plan.retry.delay_before(attempt + 1, raw, self._rng)  # type: ignore[union-attr]
                attempts.append(AttemptRecord(attempt, raw.status_code, None, elapsed, wait * 1000))
                raw.close()
                self._sleep(wait)
                continue
            attempts.append(AttemptRecord(attempt, raw.status_code, None, elapsed))
            return self._finish(plan, raw, attempts, hops, started)

    request.__doc__ = _REQUEST_DOC

    def _send(self, plan: _Plan, hops: list[str]) -> httpx.Response:
        request = self._client.build_request(plan.method, plan.url, params=plan.params, headers=plan.headers,
                                             timeout=plan.timeout, **plan.body)
        response = self._client.send(request, auth=plan.auth, follow_redirects=False)
        while (step := self._redirect(response, hops, plan)) is not None:
            next_request, auth = step
            response.close()
            response = self._client.send(next_request, auth=auth, follow_redirects=False)
        return response

    def get(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("DELETE", path, **kwargs)

    def head(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("HEAD", path, **kwargs)

    def options(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("OPTIONS", path, **kwargs)


# --------------------------------------------------------------------------- async client


class AsyncApiClient(_ClientBase):
    """Asynchronous API client with the same behaviour as :class:`ApiClient`.

    Useful for concurrent calls (``asyncio.gather``). Use ``async with`` or
    ``await aclose()``.
    """

    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, **options: Any) -> None:
        super().__init__(base_url, **options)
        self._sleep = sleep
        self._client = httpx.AsyncClient(transport=transport, **self._options)

    @property
    def httpx_client(self) -> httpx.AsyncClient:
        return self._client

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncApiClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def request(self, method: str, path: str, *, path_params: Mapping[str, Any] | None = None,
                      params: Any = None, headers: Mapping[str, str] | None = None,
                      cookies: Mapping[str, str] | None = None, json: Any = UNSET, data: Any = None,
                      files: Any = None, content: Any = None, timeout: Any = None, auth: Any = UNSET,
                      follow_redirects: bool | None = None, retry: Any = UNSET,
                      idempotency_key: str | None = None, idempotent: bool = False) -> ApiResponse:
        plan = self._plan(method, path, path_params=path_params, params=params, headers=headers,
                          cookies=cookies, json=json, data=data, files=files, content=content,
                          timeout=timeout, auth=auth, follow_redirects=follow_redirects, retry=retry,
                          idempotency_key=idempotency_key, idempotent=idempotent)
        attempts: list[AttemptRecord] = []
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            hops: list[str] = []
            began = time.perf_counter()
            try:
                raw = await self._send(plan, hops)
            except httpx.TransportError as error:
                elapsed = (time.perf_counter() - began) * 1000
                if not self._retry_error(plan, error, attempt):
                    attempts.append(AttemptRecord(attempt, None, type(error).__name__, elapsed))
                    raise self._transport_error(plan, error, attempt) from error
                wait = plan.retry.delay_before(attempt + 1, None, self._rng)  # type: ignore[union-attr]
                attempts.append(AttemptRecord(attempt, None, type(error).__name__, elapsed, wait * 1000))
                await self._sleep(wait)
                continue
            elapsed = (time.perf_counter() - began) * 1000
            if self._retry_status(plan, raw, attempt):
                wait = plan.retry.delay_before(attempt + 1, raw, self._rng)  # type: ignore[union-attr]
                attempts.append(AttemptRecord(attempt, raw.status_code, None, elapsed, wait * 1000))
                await raw.aclose()
                await self._sleep(wait)
                continue
            attempts.append(AttemptRecord(attempt, raw.status_code, None, elapsed))
            return self._finish(plan, raw, attempts, hops, started)

    request.__doc__ = _REQUEST_DOC

    async def _send(self, plan: _Plan, hops: list[str]) -> httpx.Response:
        request = self._client.build_request(plan.method, plan.url, params=plan.params, headers=plan.headers,
                                             timeout=plan.timeout, **plan.body)
        response = await self._client.send(request, auth=plan.auth, follow_redirects=False)
        while (step := self._redirect(response, hops, plan)) is not None:
            next_request, auth = step
            await response.aclose()
            response = await self._client.send(next_request, auth=auth, follow_redirects=False)
        return response

    async def get(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("PUT", path, **kwargs)

    async def patch(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("PATCH", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("DELETE", path, **kwargs)

    async def head(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("HEAD", path, **kwargs)

    async def options(self, path: str, **kwargs: Any) -> ApiResponse:
        return await self.request("OPTIONS", path, **kwargs)
