"""The adapter contract: what differs between database engines, and nothing more.

Everything engine-independent (execution, transactions, validation, reporting)
works on SQLAlchemy Core connections. An adapter only answers the questions
that genuinely differ: how to build the URL and engine options for a dialect,
which per-connection setup it needs, which features it has, how to read the
server's own identity, and how to classify its driver errors.

Adapters never translate SQL. Dialect-specific SQL stays visible in the test
that uses it, and an unsupported option or operation fails explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import DBAPIError

from complete_automation.database.errors import (
    DatabaseConfigurationError,
    ErrorKind,
    UnsupportedOperationError,
)
from complete_automation.database.settings import DatabaseSettings


@dataclass(frozen=True)
class Capabilities:
    """Features that tests may depend on. Check them instead of assuming every engine matches."""

    savepoints: bool
    transactional_ddl: bool
    """DDL can be rolled back with the surrounding transaction."""
    schemas: bool
    """Real schemas (namespaces) with a configurable search path."""
    statement_timeout: bool
    isolation_levels: tuple[str, ...]
    """Isolation levels that ``isolation_level`` may name (never ``AUTOCOMMIT``)."""
    concurrent_writers: bool
    """Several transactions can write at once. False for SQLite, so concurrency behavior differs."""
    foreign_keys_always_enforced: bool
    notes: str = ""


class DatabaseAdapter(ABC):
    dialect: ClassVar[str]
    """SQLAlchemy backend name and the value of ``modules.database.dialect``."""
    driver: ClassVar[str]
    """SQLAlchemy driver name this adapter supports."""
    capabilities: ClassVar[Capabilities]

    def __init__(self, settings: DatabaseSettings, *, read_only: bool) -> None:
        self.settings = settings
        self.read_only = read_only

    # --- configuration ---------------------------------------------------

    @abstractmethod
    def build_url(self) -> URL:
        """The SQLAlchemy URL, with the password still in it (render it with ``safe_url``)."""

    @abstractmethod
    def engine_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``create_engine``: pooling, timeouts, connect arguments."""

    def configure_engine(self, engine: Engine) -> None:  # noqa: B027 - optional hook
        """Attach per-connection setup (``PRAGMA``, session settings). Must not connect."""

    def validate(self) -> None:
        """Reject settings this engine cannot honor. Raises :class:`DatabaseConfigurationError`."""
        level = self.settings.isolation_level
        if level is not None and level.upper() not in self.capabilities.isolation_levels:
            allowed = ", ".join(self.capabilities.isolation_levels)
            raise DatabaseConfigurationError(
                f"isolation_level {level!r} is not supported by {self.dialect} (use one of: {allowed})"
            )

    # --- identity and server facts (these connect) -------------------------

    @abstractmethod
    def database_name(self, url: URL) -> str:
        """The configured database identity: the database name, or ``:memory:`` / the file path."""

    @abstractmethod
    def current_database(self, connection: Connection) -> str:
        """The database the *server* says this connection is using."""

    @abstractmethod
    def server_version(self, connection: Connection) -> str: ...

    @abstractmethod
    def foreign_keys_enforced(self, connection: Connection) -> bool: ...

    # --- errors ---------------------------------------------------------

    def classify(self, error: DBAPIError) -> ErrorKind | None:
        """Engine-specific error kind, or ``None`` to fall back to the generic mapping."""
        return None

    # --- helpers ----------------------------------------------------------

    def require(self, capability: str) -> None:
        """Raise :class:`UnsupportedOperationError` unless the boolean capability is true."""
        if not getattr(self.capabilities, capability):
            raise UnsupportedOperationError(f"{self.dialect} does not support {capability.replace('_', ' ')}")
