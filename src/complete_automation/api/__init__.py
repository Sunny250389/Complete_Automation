"""API automation module: HTTP clients, authentication, validation and fixtures.

Builds on ``complete_automation.core`` for configuration, logging, redaction,
results, reporting and artifacts. Importing this package performs no network
access. Requires the ``api`` extra (``uv sync --extra api``).
"""

from complete_automation.api.auth import (
    ApiKeyAuth,
    BearerTokenAuth,
    OAuth2ClientCredentials,
    basic_auth,
    build_auth,
)
from complete_automation.api.client import ApiClient, ApiResponse, AsyncApiClient, RetryConfig
from complete_automation.api.errors import (
    ApiError,
    ApiTimeoutError,
    ApiTransportError,
    AuthenticationError,
    ContractError,
    RedirectLoopError,
    ResponseDecodeError,
    UnsafeOperationError,
)
from complete_automation.api.settings import ApiSettings

__all__ = [
    "ApiClient",
    "ApiError",
    "ApiKeyAuth",
    "ApiResponse",
    "ApiSettings",
    "ApiTimeoutError",
    "ApiTransportError",
    "AsyncApiClient",
    "AuthenticationError",
    "BearerTokenAuth",
    "ContractError",
    "OAuth2ClientCredentials",
    "RedirectLoopError",
    "ResponseDecodeError",
    "RetryConfig",
    "UnsafeOperationError",
    "basic_auth",
    "build_auth",
]
