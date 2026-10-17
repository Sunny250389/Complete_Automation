"""API exceptions, all classified through the shared-core ``ErrorCategory``.

HTTP error statuses are never raised automatically: a 404 or 500 is a
response to assert on. These exceptions cover transport failures, unusable
responses, unsafe operations and configuration problems.
"""

from __future__ import annotations

from complete_automation.core.exceptions import ErrorCategory, FrameworkError


class ApiError(FrameworkError):
    """Base class for API module errors."""


class ApiTransportError(ApiError):
    """No HTTP response was received (connection refused, DNS, protocol error)."""

    category = ErrorCategory.INFRASTRUCTURE

    def __init__(self, message: str, *, method: str, url: str, attempts: int) -> None:
        super().__init__(message)
        self.method = method
        self.url = url
        """Sanitized URL: credentials and sensitive query values are masked."""
        self.attempts = attempts


class ApiTimeoutError(ApiTransportError):
    category = ErrorCategory.TIMEOUT


class ResponseDecodeError(ApiError, AssertionError):
    """The response body could not be decoded as expected (empty or malformed JSON).

    Also an ``AssertionError``: the system under test returned an unusable body.
    """

    category = ErrorCategory.ASSERTION


class RedirectLoopError(ApiError, AssertionError):
    category = ErrorCategory.ASSERTION


class UnsafeOperationError(ApiError):
    """A request was refused because it could change a protected system or repeat a side effect."""

    category = ErrorCategory.CONFIGURATION


class AuthenticationError(ApiError):
    """Credentials could not be obtained (for example, an OAuth2 token request failed)."""

    def __init__(self, message: str, *, category: ErrorCategory = ErrorCategory.CONFIGURATION) -> None:
        super().__init__(message)
        self.category = category


class ContractError(ApiError):
    """An API contract (OpenAPI document or JSON Schema) is missing, unreadable or invalid."""

    category = ErrorCategory.CONFIGURATION
