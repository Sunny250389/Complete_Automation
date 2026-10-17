"""Sanitized views of URLs, headers and bodies for logs, messages and artifacts.

Built on the shared-core ``Redactor``. Header and query-parameter names are
masked when the redactor considers them sensitive or when an authentication
strategy declared them (for example a custom API-key header).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import httpx

from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor

ALWAYS_SENSITIVE = frozenset(
    {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key", "api-key", "x-auth-token"}
)
DEFAULT_EXCERPT_CHARS = 500


class Sanitizer:
    def __init__(self, redactor: Redactor | None = None, sensitive_names: Iterable[str] = ()) -> None:
        self.redactor = redactor or DEFAULT_REDACTOR
        self._names = ALWAYS_SENSITIVE | {name.lower() for name in sensitive_names}

    @property
    def mask(self) -> str:
        return self.redactor.mask

    def is_sensitive(self, name: str) -> bool:
        return name.lower() in self._names or self.redactor.is_sensitive_key(name)

    def url(self, url: httpx.URL | str) -> str:
        """The URL with any password and sensitive query values replaced by the mask."""
        parts = urlsplit(str(url))
        netloc = parts.netloc
        if "@" in netloc:
            userinfo, _, host = netloc.rpartition("@")
            user = userinfo.split(":", 1)[0]
            # "user:password@" keeps the user; a bare "token@" is masked entirely.
            netloc = f"{user}:{self.mask}@{host}" if ":" in userinfo else f"{self.mask}@{host}"
        query = "&".join(self._query_item(item) for item in parts.query.split("&")) if parts.query else ""
        return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))

    def _query_item(self, item: str) -> str:
        key, sep, _ = item.partition("=")
        return f"{key}={self.mask}" if sep and self.is_sensitive(unquote_plus(key)) else item

    def headers(self, headers: Mapping[str, str] | httpx.Headers) -> dict[str, str]:
        items = headers.multi_items() if isinstance(headers, httpx.Headers) else headers.items()
        result: dict[str, str] = {}
        for name, value in items:
            shown = self.mask if self.is_sensitive(name) else self.redactor.redact_text(value)
            result[name] = f"{result[name]}, {shown}" if name in result else shown
        return result

    def body(self, content: bytes, content_type: str | None, limit: int | None = DEFAULT_EXCERPT_CHARS) -> str:
        """A redacted, truncated rendering of a body; binary content is summarised, not decoded."""
        if not content:
            return "<empty>"
        mime = (content_type or "").split(";")[0].strip().lower()
        text: str | None = None
        if mime == "application/json" or mime.endswith("+json"):
            try:  # structural redaction masks sensitive keys and redacts every string value
                text = json.dumps(self.redactor.redact(json.loads(content)), ensure_ascii=False)
            except (ValueError, UnicodeDecodeError):
                text = None  # malformed JSON: fall back to text redaction below
        elif not (
            mime.startswith("text/")
            or mime in ("application/xml", "application/x-www-form-urlencoded")
            or mime.endswith("+xml")
        ):
            return f"<{len(content)} bytes of {mime or 'unknown content type'}>"
        if text is None:
            text = self.redactor.redact_text(self._decode(content))
        if limit is not None and len(text) > limit:
            return text[:limit] + f"... <{len(text) - limit} more characters>"
        return text

    @staticmethod
    def _decode(content: bytes) -> str:
        return content.decode("utf-8", errors="replace")
