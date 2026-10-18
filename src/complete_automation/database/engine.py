"""Engine lifecycle: build the engine lazily, hand out connections, dispose cleanly.

Creating an :class:`EngineManager` validates settings and builds the URL, but
opens no connection; the SQLAlchemy engine itself is created on first use and
connects only when a connection is requested. Importing this module connects to
nothing.

Pooling: file-based SQLite and PostgreSQL use SQLAlchemy's ``QueuePool``
(``pool.size`` + ``pool.max_overflow`` connections, ``pool.timeout_seconds`` to
wait for one, ``pool_pre_ping`` to replace dead ones). In-memory SQLite uses one
shared connection. Call :meth:`dispose` (or use the manager as a context
manager) to close every pooled connection; fixtures do this at the end of the
scope that created the manager.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, event
from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import URL, Connection, Engine

from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor
from complete_automation.database.adapters import DatabaseAdapter, get_adapter_class
from complete_automation.database.classification import wrap_error
from complete_automation.database.errors import DatabaseConfigurationError, DatabaseError
from complete_automation.database.safety import OperationPolicy
from complete_automation.database.settings import DatabaseSettings, safe_url

if TYPE_CHECKING:
    from complete_automation.database.session import DatabaseSession


@dataclass(frozen=True)
class HealthReport:
    """Result of :meth:`EngineManager.health_check`. Never contains credentials."""

    ok: bool
    dialect: str
    database: str
    latency_ms: float
    server_version: str | None = None
    error_kind: str | None = None
    error: str | None = None


class EngineManager:
    def __init__(
        self,
        settings: DatabaseSettings,
        *,
        environment: str = "dev",
        redactor: Redactor = DEFAULT_REDACTOR,
    ) -> None:
        """Validate ``settings`` for ``environment``; opens no connection.

        Raises:
            DatabaseConfigurationError: invalid or inconsistent settings, a missing driver,
                or write access requested in a protected environment.
        """
        self.settings = settings
        self.environment = environment
        self.redactor = redactor
        adapter_class = get_adapter_class(settings.resolved_dialect)
        # Resolve the effective access first: a protected environment forces read-only.
        read_only = settings.access == "read_only" or settings.is_protected(environment)
        self.adapter: DatabaseAdapter = adapter_class(settings, read_only=read_only)
        self.adapter.validate()
        self.url: URL = self.adapter.build_url()
        self.policy = OperationPolicy.from_settings(
            settings, environment=environment, configured_database=self.adapter.database_name(self.url)
        )
        self._engine: Engine | None = None
        self._lock = threading.Lock()
        self._verified_database: str | None = None
        self._open = 0

    # --- identity and display ---------------------------------------------

    @property
    def dialect(self) -> str:
        return self.adapter.dialect

    @property
    def database(self) -> str:
        return self.policy.configured_database

    @property
    def redacted_url(self) -> str:
        """The connection URL with the password and sensitive query values masked."""
        return safe_url(self.url, self.redactor)

    def __repr__(self) -> str:
        return f"EngineManager({self.redacted_url!r}, read_only={self.policy.read_only})"

    # --- engine and connections -------------------------------------------

    @property
    def engine(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = self._create_engine()
            return self._engine

    def _create_engine(self) -> Engine:
        try:
            engine = create_engine(self.url, **self.adapter.engine_kwargs())
        except ImportError as error:  # driver missing at import time
            raise DatabaseConfigurationError(
                f"the {self.adapter.driver} driver for {self.dialect} could not be imported"
            ) from error
        except sa_exc.SQLAlchemyError as error:
            raise wrap_error(error, self.adapter, operation="engine creation", redactor=self.redactor) from None
        self.adapter.configure_engine(engine)

        @event.listens_for(engine, "checkout")
        def _checked_out(*_: object) -> None:
            self._open += 1

        @event.listens_for(engine, "checkin")
        def _checked_in(*_: object) -> None:
            self._open -= 1

        return engine

    @property
    def open_connections(self) -> int:
        """Connections currently checked out of the pool. Zero means nothing is leaked."""
        return self._open

    def connect(self) -> Connection:
        """Open a connection. The caller must close it; prefer :meth:`connection` or :meth:`session`."""
        try:
            return self.engine.connect()
        except sa_exc.SQLAlchemyError as error:
            raise wrap_error(error, self.adapter, operation="connect", redactor=self.redactor) from None

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """A raw SQLAlchemy connection, rolled back (if still in a transaction) and closed on exit."""
        connection = self.connect()
        try:
            yield connection
        finally:
            try:
                connection.close()  # rolls back any open transaction
            except sa_exc.SQLAlchemyError:
                pass

    def session(
        self,
        *,
        commit_on_success: bool = False,
        isolation_level: str | None = None,
        on_query: Callable[[dict[str, object]], None] | None = None,
    ) -> DatabaseSession:
        """Open a :class:`DatabaseSession`: one connection, one executor, explicit transactions.

        Nothing is committed unless the code calls ``commit()`` (or
        ``commit_on_success=True`` is given and the ``with`` block succeeds).
        """
        from complete_automation.database.session import DatabaseSession

        return DatabaseSession(
            self,
            commit_on_success=commit_on_success,
            isolation_level=isolation_level,
            on_query=on_query,
        )

    # --- identity verification ----------------------------------------------

    def verified_database(self, connection: Connection) -> str:
        """The database name the *server* reports for ``connection`` (cached after the first success)."""
        if self._verified_database is None:
            try:
                self._verified_database = self.adapter.current_database(connection)
            except sa_exc.SQLAlchemyError as error:
                raise wrap_error(error, self.adapter, operation="identity check", redactor=self.redactor) from None
        return self._verified_database

    # --- health and shutdown ---------------------------------------------------

    def health_check(self) -> HealthReport:
        """Connect, read the server version and report. Connection problems do not raise."""
        started = time.perf_counter()
        try:
            with self.connection() as connection:
                version = self.adapter.server_version(connection)
                database = self.adapter.current_database(connection)
        except DatabaseError as error:
            return self._failed_health(started, getattr(error, "kind", "connection"), str(error))
        except sa_exc.SQLAlchemyError as error:
            wrapped = wrap_error(error, self.adapter, operation="health check", redactor=self.redactor)
            return self._failed_health(started, getattr(wrapped, "kind", "connection"), str(wrapped))
        return HealthReport(
            ok=True,
            dialect=self.dialect,
            database=database,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            server_version=version,
        )

    def _failed_health(self, started: float, kind: str, message: str) -> HealthReport:
        return HealthReport(
            ok=False,
            dialect=self.dialect,
            database=self.database,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            error_kind=kind,
            error=self.redactor.redact_text(message),
        )

    def dispose(self) -> None:
        """Close every pooled connection. Safe to call twice; the manager can be reused afterwards."""
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.dispose()

    def __enter__(self) -> EngineManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.dispose()
