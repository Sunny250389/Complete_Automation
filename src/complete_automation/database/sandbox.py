"""Settings for databases a test creates and owns (temporary SQLite files, scratch PostgreSQL schemas).

A sandbox database is one the test run itself created, so granting it write,
DDL and destructive rights is safe: the ``privileged_databases`` allowlist names
exactly that database. Never build sandbox settings for a shared or
pre-existing database.
"""

from __future__ import annotations

import secrets
from pathlib import Path, PurePath

from complete_automation.database.settings import MEMORY, DatabaseSettings


def sqlite_sandbox_settings(database: str | Path = MEMORY, **overrides: object) -> DatabaseSettings:
    """Full-access settings for a SQLite file or ``:memory:`` database that the caller owns."""
    target = str(database)
    name = MEMORY if target == MEMORY else PurePath(target).name
    return DatabaseSettings(
        dialect="sqlite",
        database=target,
        access="read_write",
        allow_ddl=True,
        allow_destructive=True,
        privileged_databases=(name,),
        **overrides,  # type: ignore[arg-type]
    )


def unique_schema_name(prefix: str = "cauto_t") -> str:
    """A fresh schema name for a scratch PostgreSQL schema, safe to use as an identifier."""
    return f"{prefix}_{secrets.token_hex(6)}"
