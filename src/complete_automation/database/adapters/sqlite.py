"""SQLite adapter: runnable everywhere, no server needed.

Modes:

- **File database** (default pool): every connection sees the same file. Read-only
  mode opens it with ``mode=ro``, so SQLite itself refuses writes and never
  creates a missing file.
- **In-memory** (``:memory:``): one shared connection (``StaticPool``), because every
  new connection to ``:memory:`` would otherwise be a separate empty database.
  All sessions share that connection, so they share transaction state: use a file
  database to test commits, rollbacks and isolation across connections.

Differences that tests must not paper over: a single writer at a time (no
concurrent-writer behavior), dynamic typing (column types are affinities),
foreign keys off unless enabled (this adapter enables them by default), and no
server-side statement timeout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import event, pool, text
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import DBAPIError

from complete_automation.database.adapters.base import Capabilities, DatabaseAdapter
from complete_automation.database.errors import DatabaseConfigurationError, ErrorKind
from complete_automation.database.settings import MEMORY


class SqliteAdapter(DatabaseAdapter):
    dialect = "sqlite"
    driver = "pysqlite"
    capabilities = Capabilities(
        savepoints=True,
        transactional_ddl=True,
        schemas=False,
        statement_timeout=False,
        isolation_levels=("SERIALIZABLE", "READ UNCOMMITTED"),
        concurrent_writers=False,
        foreign_keys_always_enforced=False,
        notes="Single writer; dynamic typing; foreign keys need PRAGMA foreign_keys (enabled here).",
    )

    def validate(self) -> None:
        super().validate()
        s = self.settings
        if s.url is None and not s.database:
            raise DatabaseConfigurationError("sqlite needs modules.database.database (a file path or ':memory:')")
        for name, value in (
            ("host", s.host),
            ("port", s.port),
            ("username", s.username),
            ("password", s.password),
            ("schema", s.schema_name),
            ("tls.mode", s.tls.mode),
            ("statement_timeout_seconds", s.statement_timeout_seconds),
        ):
            if value is not None:
                raise DatabaseConfigurationError(f"{name} is not supported by sqlite")
        url = s.parsed_url()
        if url is not None and url.drivername not in ("sqlite", "sqlite+pysqlite"):
            raise DatabaseConfigurationError("the sqlite adapter supports only the pysqlite driver")

    def _target(self) -> str:
        url = self.settings.parsed_url()
        database = (url.database if url is not None else self.settings.database) or MEMORY
        return MEMORY if database in ("", MEMORY) else database

    @property
    def in_memory(self) -> bool:
        return self._target() == MEMORY

    def build_url(self) -> URL:
        target = self._target()
        if target == MEMORY:
            return URL.create("sqlite+pysqlite", database=MEMORY)
        if self.read_only:  # SQLite refuses writes and does not create a missing file
            return URL.create(
                "sqlite+pysqlite",
                database=Path(target).resolve().as_uri(),
                query={"mode": "ro", "uri": "true"},
            )
        return URL.create("sqlite+pysqlite", database=target)

    def engine_kwargs(self) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {
            "connect_args": {"timeout": s.connect_timeout_seconds, "check_same_thread": False},
            "hide_parameters": True,
        }
        if s.isolation_level is not None:
            kwargs["isolation_level"] = s.isolation_level.upper()
        if self.in_memory:
            kwargs["poolclass"] = pool.StaticPool
        else:
            kwargs.update(
                pool_size=s.pool.size,
                max_overflow=s.pool.max_overflow,
                pool_timeout=s.pool.timeout_seconds,
                pool_pre_ping=s.pool.pre_ping,
            )
            if s.pool.recycle_seconds is not None:
                kwargs["pool_recycle"] = int(s.pool.recycle_seconds)
        return kwargs

    def configure_engine(self, engine: Engine) -> None:
        settings, read_only, in_memory = self.settings, self.read_only, self.in_memory

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_connection: Any, _record: Any) -> None:
            # The driver's implicit transaction handling breaks SAVEPOINT and DDL
            # rollback; take over transaction control (SQLAlchemy's documented recipe).
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(f"PRAGMA foreign_keys = {'ON' if settings.sqlite.foreign_keys else 'OFF'}")
                if settings.sqlite.journal_mode and not in_memory and not read_only:
                    cursor.execute(f"PRAGMA journal_mode = {settings.sqlite.journal_mode.upper()}")
                if read_only:
                    cursor.execute("PRAGMA query_only = ON")
            finally:
                cursor.close()

        @event.listens_for(engine, "begin")
        def _on_begin(connection: Connection) -> None:
            connection.exec_driver_sql("BEGIN")

    def database_name(self, url: URL) -> str:
        return self._target()

    def current_database(self, connection: Connection) -> str:
        rows = connection.exec_driver_sql("PRAGMA database_list").fetchall()
        for row in rows:
            if row[1] == "main":
                return os.path.abspath(row[2]) if row[2] else MEMORY
        return MEMORY

    def server_version(self, connection: Connection) -> str:
        return str(connection.execute(text("select sqlite_version()")).scalar_one())

    def foreign_keys_enforced(self, connection: Connection) -> bool:
        return bool(connection.exec_driver_sql("PRAGMA foreign_keys").scalar())

    def classify(self, error: DBAPIError) -> ErrorKind | None:
        name = getattr(error.orig, "sqlite_errorname", "") or ""
        message = str(error.orig).lower()
        if name in ("SQLITE_BUSY", "SQLITE_LOCKED") or "database is locked" in message:
            return "timeout"
        if name in ("SQLITE_CANTOPEN", "SQLITE_NOTADB") or "unable to open database" in message:
            return "connection"
        if name in ("SQLITE_READONLY", "SQLITE_AUTH", "SQLITE_PERM") or "readonly" in message:
            return "permission"
        if name == "SQLITE_INTERRUPT":
            return "timeout"
        if "syntax error" in message or message.startswith("no such "):
            return "syntax"
        return None
