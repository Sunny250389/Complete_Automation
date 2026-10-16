"""Masking of sensitive values in structured data and free text.

Redaction is pattern-based and therefore best-effort: it catches values stored
under sensitive key names and common credential shapes in text (URL userinfo,
``Authorization`` headers, bearer tokens, ``key=value`` pairs with a sensitive
key, PEM private keys). It cannot recognise a secret that has neither a
sensitive key nor a recognisable shape, so modules must still avoid logging
raw payloads.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, SecretBytes, SecretStr

DEFAULT_MASK = "***"

# Whole key names (normalised: lower case, "-" and " " become "_").
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "x_api_key",
        "access_key",
        "private_key",
        "secret_key",
        "client_secret",
        "connection_string",
        "conn_str",
        "proxy_authorization",
        "set_cookie",
    }
)
# Any key containing one of these as a "_"-separated segment.
_SENSITIVE_SEGMENTS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "passphrase",
        "secret",
        "token",
        "authorization",
        "auth",
        "cookie",
        "credential",
        "credentials",
    }
)

_KEY_WORDS = (
    r"password|passwd|pwd|passphrase|secret|token|credentials?"
    r"|api[_-]?key|apikey|access[_-]?key|private[_-]?key|client[_-]?secret"
)
_VALUE = r"""(?:"[^"]*"|'[^']*'|[^\s,;&}\]"']+)"""

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # PEM private key blocks.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
        ),
        "{mask}",
    ),
    # scheme://user:password@host -> scheme://user:***@host
    (
        re.compile(r"(?P<head>\b[a-zA-Z][a-zA-Z0-9+.-]*://[^:/@\s]+:)(?P<secret>[^@\s/]+)(?=@)"),
        r"\g<head>{mask}",
    ),
    # Authorization: Bearer abc / Authorization=abc
    (
        re.compile(
            r"(?i)(?P<head>\b(?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?(?:(?:bearer|basic|token|digest)\s+)?)"
            r"(?P<secret>[^\s,;\"']+)"
        ),
        r"\g<head>{mask}",
    ),
    # Bare "Bearer <token>".
    (re.compile(r"(?i)(?P<head>\bbearer\s+)(?P<secret>[A-Za-z0-9._~+/-]+=*)"), r"\g<head>{mask}"),
    # key=value, key: value, "key": "value" where the key looks sensitive.
    (
        re.compile(
            rf"(?i)(?P<head>[\"']?\b[A-Za-z0-9_.-]*?(?:{_KEY_WORDS})(?:[_.-][A-Za-z0-9_.-]*)?[\"']?\s*[:=]\s*)"
            rf"(?P<secret>{_VALUE})"
        ),
        r"\g<head>{mask}",
    ),
)


def _normalise_key(key: str) -> str:
    return re.sub(r"[\s\-.]+", "_", key.strip().lower())


class Redactor:
    """Masks sensitive values. Immutable once built and safe to share."""

    def __init__(self, extra_keys: Iterable[str] = (), mask: str = DEFAULT_MASK) -> None:
        if not mask:
            raise ValueError("mask must not be empty")
        self.mask = mask
        self._extra_keys = frozenset(_normalise_key(k) for k in extra_keys)
        self._patterns = tuple(
            (pattern, replacement.replace("{mask}", mask.replace("\\", "\\\\")))
            for pattern, replacement in _PATTERNS
        )

    def is_sensitive_key(self, key: str) -> bool:
        normalised = _normalise_key(key)
        if normalised in _SENSITIVE_KEYS or normalised in self._extra_keys:
            return True
        return any(segment in _SENSITIVE_SEGMENTS for segment in normalised.split("_"))

    def redact_text(self, text: str) -> str:
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return text

    def redact(self, value: Any) -> Any:
        """Return a redacted copy of ``value``; containers are copied, never mutated."""
        return self._redact(value, set())

    def _redact(self, value: Any, seen: set[int]) -> Any:
        if isinstance(value, (SecretStr, SecretBytes)):
            return self.mask
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, BaseModel):
            return self._redact(value.model_dump(mode="json"), seen)
        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            marker = id(value)
            if marker in seen:
                return "<cycle>"
            seen = seen | {marker}
            if isinstance(value, Mapping):
                return {
                    key: (
                        self.mask
                        if isinstance(key, str) and item is not None and self.is_sensitive_key(key)
                        else self._redact(item, seen)
                    )
                    for key, item in value.items()
                }
            items = [self._redact(item, seen) for item in value]
            return tuple(items) if isinstance(value, tuple) else items
        return value


DEFAULT_REDACTOR = Redactor()
