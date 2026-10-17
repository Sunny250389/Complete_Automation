"""Authentication strategies, implemented as ``httpx.Auth`` flows.

``httpx.Auth`` is the extension point: a new strategy subclasses it and works
with both clients without changes to the transport layer. Strategies that put
credentials in custom headers or query parameters expose them through
``sensitive_names`` so diagnostics and cross-origin redirects never leak them.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Literal

import httpx
from pydantic import SecretStr

from complete_automation.api.errors import AuthenticationError
from complete_automation.api.settings import (
    ApiKeyAuthSettings,
    AuthSettings,
    BasicAuthSettings,
    BearerAuthSettings,
    OAuth2ClientCredentialsSettings,
)
from complete_automation.core.exceptions import ErrorCategory


def _secret(value: str | SecretStr) -> str:
    return value.get_secret_value() if isinstance(value, SecretStr) else value


class ApiKeyAuth(httpx.Auth):
    """An API key in a header (preferred) or, when an API requires it, a query parameter."""

    def __init__(
        self, value: str | SecretStr, *, name: str = "X-API-Key", location: Literal["header", "query"] = "header"
    ) -> None:
        if location not in ("header", "query"):
            raise ValueError("location must be 'header' or 'query'")
        if not _secret(value):
            raise ValueError("API key must not be empty")
        self._value = SecretStr(_secret(value))
        self.name = name
        self.location = location

    @property
    def sensitive_names(self) -> tuple[str, ...]:
        return (self.name,)

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        value = self._value.get_secret_value()
        if self.location == "header":
            request.headers[self.name] = value
        else:
            request.url = request.url.copy_merge_params({self.name: value})
        yield request

    def __repr__(self) -> str:
        return f"ApiKeyAuth(name={self.name!r}, location={self.location!r}, value='***')"


class BearerTokenAuth(httpx.Auth):
    """``Authorization: Bearer <token>`` from a fixed token or a provider callable.

    ``refresh`` (optional) is called once when a response is 401; the request is
    then repeated with the new token. Providers and refresh hooks are plain
    callables and are used by both the sync and async clients.
    """

    def __init__(
        self,
        token: str | SecretStr | None = None,
        *,
        token_provider: Callable[[], str] | None = None,
        refresh: Callable[[], str] | None = None,
    ) -> None:
        if (token is None) == (token_provider is None):
            raise ValueError("give exactly one of token or token_provider")
        self._token = SecretStr(_secret(token)) if token is not None else None
        self._provider = token_provider
        self._refresh = refresh

    def _current(self) -> str:
        return self._token.get_secret_value() if self._token is not None else self._provider()  # type: ignore[misc]

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._current()}"
        response = yield request
        if response.status_code == 401 and self._refresh is not None:
            self._token = SecretStr(self._refresh())
            request.headers["Authorization"] = f"Bearer {self._current()}"
            yield request

    def __repr__(self) -> str:
        return "BearerTokenAuth(token='***')"


def basic_auth(username: str, password: str | SecretStr) -> httpx.BasicAuth:
    """HTTP Basic authentication, encoded by httpx."""
    return httpx.BasicAuth(username, _secret(password))


class OAuth2ClientCredentials(httpx.Auth):
    """OAuth2 client-credentials grant (RFC 6749 section 4.4) with token caching.

    The token is cached per instance until ``expires_in`` minus ``leeway_seconds``
    (no ``expires_in``: until a 401). A 401 from the API discards the token and
    retries the request once with a fresh one. Errors name the token endpoint,
    the HTTP status and the OAuth ``error`` code, never credentials or tokens.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str | SecretStr,
        *,
        scopes: tuple[str, ...] | list[str] = (),
        audience: str | None = None,
        client_auth: Literal["basic", "body"] = "basic",
        leeway_seconds: float = 30.0,
        timeout: float = 10.0,
        verify: bool = True,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if client_auth not in ("basic", "body"):
            raise ValueError("client_auth must be 'basic' or 'body'")
        self.token_url = token_url
        self.client_id = client_id
        self._client_secret = SecretStr(_secret(client_secret))
        self.scopes = tuple(scopes)
        self.audience = audience
        self.client_auth = client_auth
        self.leeway_seconds = leeway_seconds
        self._timeout = timeout
        self._verify = verify
        self._transport = transport
        self._async_transport = async_transport
        self._clock = clock
        self._token: SecretStr | None = None
        self._expires_at: float | None = None
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None
        self.tokens_issued = 0

    # -- token request -----------------------------------------------------------------

    def _token_request(self) -> tuple[dict[str, str], httpx.Auth | httpx._client.UseClientDefault]:
        data = {"grant_type": "client_credentials"}
        if self.scopes:
            data["scope"] = " ".join(self.scopes)
        if self.audience:
            data["audience"] = self.audience
        if self.client_auth == "body":
            data.update(client_id=self.client_id, client_secret=self._client_secret.get_secret_value())
            return data, httpx.USE_CLIENT_DEFAULT  # the token client has no auth of its own
        return data, httpx.BasicAuth(self.client_id, self._client_secret.get_secret_value())

    def _accept(self, response: httpx.Response) -> None:
        if response.status_code != 200:
            code = ""
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("error"), str):
                    code = f" ({body['error']})"
            except ValueError:
                pass
            category = ErrorCategory.INFRASTRUCTURE if response.status_code >= 500 else ErrorCategory.CONFIGURATION
            raise AuthenticationError(
                f"OAuth2 token request to {self.token_url} failed: HTTP {response.status_code}{code}",
                category=category,
            )
        try:
            body = response.json()
        except ValueError:
            raise AuthenticationError(f"OAuth2 token response from {self.token_url} is not JSON") from None
        token = body.get("access_token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise AuthenticationError(f"OAuth2 token response from {self.token_url} has no access_token")
        token_type = body.get("token_type", "Bearer")
        if not isinstance(token_type, str) or token_type.lower() != "bearer":
            raise AuthenticationError(f"unsupported OAuth2 token_type from {self.token_url}")
        expires_in = body.get("expires_in")
        self._token = SecretStr(token)
        self._expires_at = (
            self._clock() + max(0.0, float(expires_in) - self.leeway_seconds)
            if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool)
            else None
        )
        self.tokens_issued += 1

    def _valid_token(self) -> str | None:
        if self._token is None:
            return None
        if self._expires_at is not None and self._clock() >= self._expires_at:
            return None
        return self._token.get_secret_value()

    def invalidate(self) -> None:
        self._token, self._expires_at = None, None

    def _transport_failure(self, error: httpx.TransportError) -> AuthenticationError:
        return AuthenticationError(
            f"OAuth2 token request to {self.token_url} failed: {type(error).__name__}",
            category=ErrorCategory.INFRASTRUCTURE,
        )

    def token(self) -> str:
        with self._lock:
            current = self._valid_token()
            if current is not None:
                return current
            data, auth = self._token_request()
            try:
                with httpx.Client(
                    transport=self._transport, timeout=self._timeout, verify=self._verify, trust_env=False
                ) as client:
                    response = client.post(self.token_url, data=data, auth=auth, headers={"Accept": "application/json"})
            except httpx.TransportError as error:
                raise self._transport_failure(error) from error
            self._accept(response)
            return self._token.get_secret_value()  # type: ignore[union-attr]

    async def async_token(self) -> str:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            current = self._valid_token()
            if current is not None:
                return current
            data, auth = self._token_request()
            try:
                async with httpx.AsyncClient(
                    transport=self._async_transport, timeout=self._timeout, verify=self._verify, trust_env=False
                ) as client:
                    response = await client.post(
                        self.token_url, data=data, auth=auth, headers={"Accept": "application/json"}
                    )
            except httpx.TransportError as error:
                raise self._transport_failure(error) from error
            self._accept(response)
            return self._token.get_secret_value()  # type: ignore[union-attr]

    # -- httpx flows -------------------------------------------------------------------

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token()}"
        response = yield request
        if response.status_code == 401:
            self.invalidate()
            request.headers["Authorization"] = f"Bearer {self.token()}"
            yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = f"Bearer {await self.async_token()}"
        response = yield request
        if response.status_code == 401:
            self.invalidate()
            request.headers["Authorization"] = f"Bearer {await self.async_token()}"
            yield request

    def __repr__(self) -> str:
        return f"OAuth2ClientCredentials(token_url={self.token_url!r}, client_id={self.client_id!r}, client_secret='***')"


def build_auth(settings: AuthSettings | None) -> httpx.Auth | None:
    """Create the strategy described by ``modules.api.auth``."""
    if settings is None:
        return None
    if isinstance(settings, ApiKeyAuthSettings):
        return ApiKeyAuth(settings.value, name=settings.name, location=settings.location)
    if isinstance(settings, BearerAuthSettings):
        return BearerTokenAuth(settings.token)
    if isinstance(settings, BasicAuthSettings):
        return basic_auth(settings.username, settings.password)
    if isinstance(settings, OAuth2ClientCredentialsSettings):
        return OAuth2ClientCredentials(
            settings.token_url,
            settings.client_id,
            settings.client_secret,
            scopes=settings.scopes,
            audience=settings.audience,
            client_auth=settings.client_auth,
            leeway_seconds=settings.leeway_seconds,
        )
    raise TypeError(f"unsupported auth settings: {type(settings).__name__}")
