"""Settings for the database module, read from ``modules.database`` of the shared configuration.

Example (``config/environments/<env>.yaml``)::

    modules:
      database:
        dialect: postgresql
        host: db.example.test
        port: 5432
        database: app_test
        username: app_tester        # password: CAUTO_MODULES__DATABASE__PASSWORD
        access: read_only           # the default; writes need read_write

Secrets belong in ``CAUTO_MODULES__DATABASE__...`` environment variables or
``.env``, never in YAML files. The defaults are conservative: read-only, no DDL,
no destructive statements. Application-level checks are a guardrail; they do not
replace least-privilege database credentials.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor
from complete_automation.database.errors import DatabaseConfigurationError

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")
MEMORY = ":memory:"

Dialect = Literal["sqlite", "postgresql"]
SslMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class PoolSettings(_Section):
    """Connection pool. SQLite in-memory databases ignore it (one shared connection)."""

    size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    timeout_seconds: float = Field(default=30.0, gt=0)
    """How long to wait for a free pooled connection."""
    recycle_seconds: float | None = Field(default=1800.0, gt=0)
    pre_ping: bool = True


class TlsSettings(_Section):
    mode: SslMode | None = None
    ca_bundle: Path | None = None
    """Root certificate file for ``verify-ca`` / ``verify-full``."""


class SqliteSettings(_Section):
    foreign_keys: bool = True
    """SQLite ignores foreign keys unless ``PRAGMA foreign_keys`` is on; it is on by default here."""
    journal_mode: Literal["delete", "truncate", "persist", "memory", "wal", "off"] | None = None


class DatabaseSettings(_Section):
    dialect: Dialect | None = None
    """Inferred from ``url`` when that is set; otherwise required for a non-empty section."""
    url: SecretStr | None = None
    """Full SQLAlchemy URL. Mutually exclusive with host/port/username/password."""
    database: str | None = None
    """PostgreSQL database name, or the SQLite file path (``:memory:`` for an in-memory database)."""
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    username: str | None = None
    password: SecretStr | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    """PostgreSQL schema put first on the ``search_path`` of every connection."""
    pool: PoolSettings = PoolSettings()
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    statement_timeout_seconds: float | None = Field(default=None, gt=0)
    """Server-side statement timeout where the engine supports one (PostgreSQL)."""
    isolation_level: str | None = None
    tls: TlsSettings = TlsSettings()
    sqlite: SqliteSettings = SqliteSettings()

    access: Literal["read_only", "read_write"] = "read_only"
    allow_ddl: bool = False
    allow_destructive: bool = False
    """DROP, TRUNCATE and DELETE without WHERE. Needs ``privileged_databases``."""
    privileged_databases: tuple[str, ...] = ()
    """Database names (file name for SQLite) on which DDL and destructive statements may run."""
    protected_environments: tuple[str, ...] = ("prod", "production")
    """In these environments the connection is read-only and cannot be made writable."""

    max_rows: int = Field(default=100_000, ge=1)
    """Rows a non-streaming query may return before :class:`ResultTooLargeError`."""
    capture_diagnostics: Literal["never", "on_failure", "always"] = "on_failure"

    @field_validator("database", "host", "username", "schema_name")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("privileged_databases")
    @classmethod
    def _no_blank_allowlist_entries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not entry.strip() for entry in value):
            raise ValueError("entries must not be blank")
        return value

    @field_validator("schema_name")
    @classmethod
    def _schema_is_identifier(cls, value: str | None) -> str | None:
        if value is not None and not IDENTIFIER.match(value):
            raise ValueError("must be a plain identifier (letters, digits, '_' or '$')")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> DatabaseSettings:
        parts = (self.host, self.port, self.username, self.password)
        if self.url is not None:
            if any(part is not None for part in parts) or self.database is not None:
                raise ValueError("url cannot be combined with host, port, database, username or password")
            self._parsed_url()  # validate now, so a bad URL fails at configuration time
        if (self.allow_ddl or self.allow_destructive) and not self.privileged_databases:
            raise ValueError("allow_ddl and allow_destructive require privileged_databases")
        if (self.allow_ddl or self.allow_destructive) and self.access != "read_write":
            raise ValueError("allow_ddl and allow_destructive require access: read_write")
        return self

    @property
    def is_configured(self) -> bool:
        return self.url is not None or self.database is not None or self.host is not None

    def _parsed_url(self) -> URL:
        assert self.url is not None
        try:
            return make_url(self.url.get_secret_value())
        except ArgumentError:
            # The offending text is not echoed: it may contain the password.
            raise ValueError("url is not a valid SQLAlchemy URL") from None

    @property
    def resolved_dialect(self) -> Dialect:
        """The dialect from ``dialect``, else from ``url``."""
        if self.dialect is not None:
            return self.dialect
        if self.url is not None:
            backend = self._parsed_url().get_backend_name()
            if backend in ("sqlite", "postgresql"):
                return backend  # type: ignore[return-value]
            raise DatabaseConfigurationError(f"unsupported database engine in url: {backend!r}")
        raise DatabaseConfigurationError("modules.database.dialect is not set")

    def parsed_url(self) -> URL | None:
        return self._parsed_url() if self.url is not None else None

    def with_base_dir(self, base_dir: Path) -> DatabaseSettings:
        """Resolve a relative SQLite file path and TLS bundle against ``base_dir``."""
        updates: dict[str, object] = {}
        if (
            self.url is None
            and self.database
            and self.dialect == "sqlite"
            and self.database != MEMORY
            and not Path(self.database).is_absolute()
        ):
            updates["database"] = str((base_dir / self.database).resolve())
        ca = self.tls.ca_bundle
        if ca is not None and not ca.is_absolute():
            updates["tls"] = self.tls.model_copy(update={"ca_bundle": (base_dir / ca).resolve()})
        return self.model_copy(update=updates) if updates else self

    def is_protected(self, environment: str) -> bool:
        return environment.lower() in {name.lower() for name in self.protected_environments}


def safe_url(url: URL, redactor: Redactor = DEFAULT_REDACTOR) -> str:
    """Render ``url`` with the password hidden and sensitive query values masked."""
    query = {
        key: (redactor.mask if redactor.is_sensitive_key(key) else value)
        for key, value in url.query.items()
    }
    return url.set(query=query).render_as_string(hide_password=True)
