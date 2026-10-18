"""PostgreSQL adapter (psycopg 3 through SQLAlchemy).

Needs the ``postgresql`` extra (``uv sync --extra postgresql``); the driver is
imported only when an engine is created, never at import time.

Read-only mode sets ``default_transaction_read_only`` on every connection, so the
server rejects writes even if application-level checks are bypassed. Even so,
use an account with only the privileges the tests need. ``schema`` becomes the
connection's ``search_path``; a ``statement_timeout`` is applied server-side.

Unlike SQLite, PostgreSQL has transactional DDL, real concurrent writers and
always-enforced foreign keys, so tests of those behaviors belong here.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.exc import DBAPIError

from complete_automation.database.adapters.base import Capabilities, DatabaseAdapter
from complete_automation.database.errors import DatabaseConfigurationError, ErrorKind, sqlstate_of

_ISOLATION_LEVELS = ("READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE")
_APPLICATION_NAME = "complete_automation"


class PostgreSqlAdapter(DatabaseAdapter):
    dialect = "postgresql"
    driver = "psycopg"
    capabilities = Capabilities(
        savepoints=True,
        transactional_ddl=True,
        schemas=True,
        statement_timeout=True,
        isolation_levels=_ISOLATION_LEVELS,
        concurrent_writers=True,
        foreign_keys_always_enforced=True,
        notes="READ UNCOMMITTED behaves as READ COMMITTED.",
    )

    def validate(self) -> None:
        super().validate()
        s = self.settings
        if s.url is None and not s.database:
            raise DatabaseConfigurationError("postgresql needs modules.database.database (the database name)")
        url = s.parsed_url()
        if url is not None and url.drivername not in ("postgresql", "postgresql+psycopg"):
            raise DatabaseConfigurationError("the postgresql adapter supports only the psycopg (3) driver: use postgresql+psycopg://")
        if importlib.util.find_spec("psycopg") is None:
            raise DatabaseConfigurationError("psycopg is not installed: run 'uv sync --extra postgresql'")

    def build_url(self) -> URL:
        s = self.settings
        url = s.parsed_url()
        if url is not None:
            return url.set(drivername="postgresql+psycopg")
        return URL.create(
            "postgresql+psycopg",
            username=s.username,
            password=s.password.get_secret_value() if s.password else None,
            host=s.host,
            port=s.port,
            database=s.database,
        )

    def engine_kwargs(self) -> dict[str, Any]:
        s = self.settings
        connect_args: dict[str, Any] = {
            "connect_timeout": max(1, math.ceil(s.connect_timeout_seconds)),
            "application_name": _APPLICATION_NAME,
        }
        if s.tls.mode is not None:
            connect_args["sslmode"] = s.tls.mode
        if s.tls.ca_bundle is not None:
            connect_args["sslrootcert"] = str(s.tls.ca_bundle)
        options: list[str] = []
        if self.read_only:
            options.append("-c default_transaction_read_only=on")
        if s.statement_timeout_seconds is not None:
            options.append(f"-c statement_timeout={int(s.statement_timeout_seconds * 1000)}")
        if s.schema_name is not None:
            options.append(f"-c search_path={s.schema_name}")
        if options:
            connect_args["options"] = " ".join(options)
        kwargs: dict[str, Any] = {
            "connect_args": connect_args,
            "hide_parameters": True,
            "pool_size": s.pool.size,
            "max_overflow": s.pool.max_overflow,
            "pool_timeout": s.pool.timeout_seconds,
            "pool_pre_ping": s.pool.pre_ping,
        }
        if s.pool.recycle_seconds is not None:
            kwargs["pool_recycle"] = int(s.pool.recycle_seconds)
        if s.isolation_level is not None:
            kwargs["isolation_level"] = s.isolation_level.upper()
        return kwargs

    def database_name(self, url: URL) -> str:
        return url.database or ""

    def current_database(self, connection: Connection) -> str:
        return str(connection.execute(text("select current_database()")).scalar_one())

    def server_version(self, connection: Connection) -> str:
        return str(connection.execute(text("show server_version")).scalar_one())

    def foreign_keys_enforced(self, connection: Connection) -> bool:
        return True

    def classify(self, error: DBAPIError) -> ErrorKind | None:
        state = sqlstate_of(error)
        if state is None:
            return None
        if state == "40P01":
            return "deadlock"
        if state == "40001":
            return "serialization"
        if state in ("57014", "55P03"):  # query_canceled (statement_timeout), lock_not_available
            return "timeout"
        if state.startswith(("08", "28", "53", "57P")):  # connection, auth, resources, shutdown
            return "connection"
        if state == "42501":
            return "permission"
        if state.startswith("23"):
            return "integrity"
        if state.startswith("22"):
            return "data"
        if state.startswith("42"):
            return "syntax"
        if state == "25006":  # read_only_sql_transaction
            return "permission"
        return None
