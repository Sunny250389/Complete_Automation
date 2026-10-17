"""Settings for the API module, read from ``modules.api`` of the shared configuration.

Example (``config/environments/<env>.yaml``)::

    modules:
      api:
        base_url: https://api.example.test/v1
        timeouts: {connect: 5, read: 30}
        auth:
          type: bearer      # token comes from CAUTO_MODULES__API__AUTH__TOKEN

Secrets belong in ``CAUTO_MODULES__API__...`` environment variables or ``.env``,
never in YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TimeoutSettings(_Section):
    """Finite timeouts in seconds, mapped to ``httpx.Timeout``."""

    connect: float = Field(default=5.0, gt=0)
    read: float = Field(default=30.0, gt=0)
    write: float = Field(default=30.0, gt=0)
    pool: float = Field(default=5.0, gt=0)


class ApiRetrySettings(_Section):
    """Client-level retries. Off unless ``enabled`` is true."""

    enabled: bool = False
    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=0.5, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    max_delay_seconds: float = Field(default=10.0, ge=0)
    jitter_ratio: float = Field(default=0.0, ge=0, le=1)
    status_codes: tuple[int, ...] = (429, 502, 503, 504)
    methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    """Methods retried without an idempotency key. POST and PATCH are never listed by default."""
    respect_retry_after: bool = True
    max_retry_after_seconds: float = Field(default=30.0, ge=0)


class ApiKeyAuthSettings(_Section):
    type: Literal["api_key"]
    value: SecretStr
    name: str = "X-API-Key"
    location: Literal["header", "query"] = "header"


class BearerAuthSettings(_Section):
    type: Literal["bearer"]
    token: SecretStr


class BasicAuthSettings(_Section):
    type: Literal["basic"]
    username: str
    password: SecretStr


class OAuth2ClientCredentialsSettings(_Section):
    type: Literal["oauth2_client_credentials"]
    token_url: str
    client_id: str
    client_secret: SecretStr
    scopes: tuple[str, ...] = ()
    audience: str | None = None
    client_auth: Literal["basic", "body"] = "basic"
    leeway_seconds: float = Field(default=30.0, ge=0)


AuthSettings = Annotated[
    ApiKeyAuthSettings | BearerAuthSettings | BasicAuthSettings | OAuth2ClientCredentialsSettings,
    Field(discriminator="type"),
]


class ContractSettings(_Section):
    spec_path: Path | None = None
    """Local OpenAPI document; relative paths resolve against the project base directory."""
    spec_url: str | None = None
    """Remote OpenAPI document; fetched only when set, with ``remote_timeout_seconds``."""
    remote_timeout_seconds: float = Field(default=10.0, gt=0)


class ApiSettings(_Section):
    base_url: str | None = None
    timeouts: TimeoutSettings = TimeoutSettings()
    verify_tls: bool = True
    allow_insecure_tls: bool = False
    """Must also be true to set ``verify_tls: false``; use only for local test services."""
    ca_bundle: Path | None = None
    follow_redirects: bool = False
    max_redirects: int = Field(default=5, ge=0, le=20)
    proxy: SecretStr | None = None
    default_headers: dict[str, str] = {}
    correlation_header: str | None = "X-Correlation-ID"
    latency_threshold_ms: float | None = Field(default=None, gt=0)
    retry: ApiRetrySettings = ApiRetrySettings()
    auth: AuthSettings | None = None
    read_only: bool | None = None
    """Refuse POST, PUT, PATCH and DELETE. ``None`` means: read-only in protected environments."""
    protected_environments: tuple[str, ...] = ("prod", "production")
    capture_exchanges: Literal["never", "on_failure", "always"] = "on_failure"
    log_bodies: bool = False
    contract: ContractSettings = ContractSettings()

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.lower().startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value

    @model_validator(mode="after")
    def _tls_override_is_explicit(self) -> ApiSettings:
        if not self.verify_tls and not self.allow_insecure_tls:
            raise ValueError("verify_tls: false also requires allow_insecure_tls: true")
        return self

    def is_read_only(self, environment: str) -> bool:
        if self.read_only is not None:
            return self.read_only
        return environment.lower() in {name.lower() for name in self.protected_environments}

    def with_base_dir(self, base_dir: Path) -> ApiSettings:
        """Resolve relative file paths (CA bundle, contract) against ``base_dir``."""

        def absolute(path: Path | None) -> Path | None:
            return path if path is None or path.is_absolute() else (base_dir / path).resolve()

        return self.model_copy(
            update={
                "ca_bundle": absolute(self.ca_bundle),
                "contract": self.contract.model_copy(
                    update={"spec_path": absolute(self.contract.spec_path)}
                ),
            }
        )
