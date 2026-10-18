"""Error wrapping and classification, adapter configuration, and the adapter registry. No connections."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import exc as sa_exc

from complete_automation.core.exceptions import ErrorCategory, classify_exception
from complete_automation.core.redaction import Redactor
from complete_automation.database.adapters import (
    DatabaseAdapter,
    PostgreSqlAdapter,
    SqliteAdapter,
    get_adapter_class,
    register_adapter,
)
from complete_automation.database.classification import wrap_error
from complete_automation.database.errors import (
    DataAssertionError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseIntegrityError,
    QueryExecutionError,
    QueryTimeoutError,
    SchemaValidationError,
    UnsafeOperationError,
    safe_driver_message,
)
from complete_automation.database.settings import DatabaseSettings

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]

SECRET_VALUE = "alice.private@example.test"


class _Diag:
    message_primary = 'duplicate key value violates unique constraint "customers_email_key"'


class _PgError(Exception):
    """Looks like a psycopg error: SQLSTATE, diag, and a message that leaks the key value in DETAIL."""

    def __init__(self, sqlstate: str) -> None:
        super().__init__(f'{_Diag.message_primary}\nDETAIL:  Key (email)=({SECRET_VALUE}) already exists.')
        self.sqlstate = sqlstate
        self.diag = _Diag()


def _error(cls: type[sa_exc.DBAPIError], orig: Exception, **kwargs: Any) -> sa_exc.DBAPIError:
    return cls("INSERT INTO customers (email) VALUES (:email)", {"email": SECRET_VALUE}, orig, **kwargs)


def _pg() -> PostgreSqlAdapter:
    return PostgreSqlAdapter(DatabaseSettings(dialect="postgresql", database="d"), read_only=False)


def test_driver_messages_keep_the_first_line_and_drop_detail_sql_and_values() -> None:
    wrapped = wrap_error(_error(sa_exc.IntegrityError, _PgError("23505")), _pg(), operation="INSERT")

    assert isinstance(wrapped, DatabaseIntegrityError) and wrapped.kind == "integrity" and wrapped.sqlstate == "23505"
    assert "customers_email_key" in str(wrapped) and "INSERT" in str(wrapped)
    assert SECRET_VALUE not in str(wrapped) and "DETAIL" not in str(wrapped) and "VALUES" not in str(wrapped)
    assert wrapped.category is ErrorCategory.ASSERTION


def test_the_original_exception_is_kept() -> None:
    original = _error(sa_exc.IntegrityError, _PgError("23505"))

    assert wrap_error(original, _pg()).original is original


def test_message_is_redacted_and_truncated() -> None:
    error = _error(sa_exc.OperationalError, Exception("failed for password=hunter2 " + "x" * 1000))

    message = safe_driver_message(error, Redactor())

    assert "hunter2" not in message and len(message) <= 300


@pytest.mark.parametrize(
    ("sqlstate", "kind", "category"),
    [
        ("40P01", "deadlock", ErrorCategory.INFRASTRUCTURE),
        ("40001", "serialization", ErrorCategory.INFRASTRUCTURE),
        ("57014", "timeout", ErrorCategory.TIMEOUT),
        ("55P03", "timeout", ErrorCategory.TIMEOUT),
        ("08006", "connection", ErrorCategory.INFRASTRUCTURE),
        ("28P01", "connection", ErrorCategory.INFRASTRUCTURE),
        ("42501", "permission", ErrorCategory.CONFIGURATION),
        ("25006", "permission", ErrorCategory.CONFIGURATION),
        ("23503", "integrity", ErrorCategory.ASSERTION),
        ("22001", "data", ErrorCategory.TEST_DATA),
        ("42P01", "syntax", ErrorCategory.UNKNOWN),
    ],
)
def test_postgresql_sqlstates_map_to_kinds_and_categories(sqlstate: str, kind: str, category: ErrorCategory) -> None:
    wrapped = wrap_error(_error(sa_exc.OperationalError, _PgError(sqlstate)), _pg())

    assert getattr(wrapped, "kind", "connection") == kind
    assert classify_exception(wrapped) is category


def test_deadlocks_and_serialization_failures_are_surfaced_not_hidden() -> None:
    for state in ("40P01", "40001"):
        wrapped = wrap_error(_error(sa_exc.OperationalError, _PgError(state)), _pg())

        assert isinstance(wrapped, QueryExecutionError) and wrapped.original is not None
        assert not hasattr(wrapped, "retry")  # nothing in the module retries a transaction


def test_connection_failures_become_connection_errors() -> None:
    refused = _error(sa_exc.OperationalError, Exception("connection refused"))
    invalidated = _error(sa_exc.OperationalError, Exception("server closed"), connection_invalidated=True)

    for error in (refused, invalidated):
        wrapped = wrap_error(error, None)
        assert isinstance(wrapped, DatabaseConnectionError) and wrapped.original is error
        assert classify_exception(wrapped) is ErrorCategory.INFRASTRUCTURE


def test_pool_timeouts_are_timeouts() -> None:
    wrapped = wrap_error(sa_exc.TimeoutError("QueuePool limit of size 5 overflow 10 reached"), None)

    assert isinstance(wrapped, QueryTimeoutError) and wrapped.kind == "timeout"


def test_sqlite_errors_are_classified_by_name_and_message() -> None:
    adapter = SqliteAdapter(DatabaseSettings(dialect="sqlite", database=":memory:"), read_only=False)

    class Locked(Exception):
        sqlite_errorname = "SQLITE_BUSY"

    assert wrap_error(_error(sa_exc.OperationalError, Locked("database is locked")), adapter).kind == "timeout"  # type: ignore[attr-defined]
    assert wrap_error(_error(sa_exc.OperationalError, Exception("attempt to write a readonly database")), adapter).kind == "permission"  # type: ignore[attr-defined]
    assert wrap_error(_error(sa_exc.OperationalError, Exception("no such table: x")), adapter).kind == "syntax"  # type: ignore[attr-defined]
    assert wrap_error(_error(sa_exc.IntegrityError, Exception("UNIQUE constraint failed: t.a")), adapter).kind == "integrity"  # type: ignore[attr-defined]


def test_error_categories_follow_the_shared_taxonomy() -> None:
    assert classify_exception(DataAssertionError("x")) is ErrorCategory.ASSERTION
    assert isinstance(DataAssertionError("x"), AssertionError) and isinstance(SchemaValidationError("x"), AssertionError)
    assert classify_exception(UnsafeOperationError("x")) is ErrorCategory.CONFIGURATION
    assert classify_exception(DatabaseConfigurationError("x")) is ErrorCategory.CONFIGURATION


# --- adapters ---------------------------------------------------------------------------------------------


def test_sqlite_urls_for_memory_file_and_read_only(tmp_path: Path) -> None:
    memory = SqliteAdapter(DatabaseSettings(dialect="sqlite", database=":memory:"), read_only=False)
    file = SqliteAdapter(DatabaseSettings(dialect="sqlite", database=str(tmp_path / "a.db")), read_only=False)
    ro = SqliteAdapter(DatabaseSettings(dialect="sqlite", database=str(tmp_path / "a.db")), read_only=True)

    assert memory.build_url().database == ":memory:" and memory.in_memory
    assert file.build_url().database == str(tmp_path / "a.db") and not file.in_memory
    assert ro.build_url().query == {"mode": "ro", "uri": "true"} and ro.build_url().database.startswith("file:")


def test_sqlite_memory_uses_one_shared_connection_and_files_use_a_pool(tmp_path: Path) -> None:
    from sqlalchemy import pool

    memory = SqliteAdapter(DatabaseSettings(dialect="sqlite", database=":memory:"), read_only=False).engine_kwargs()
    file = SqliteAdapter(DatabaseSettings(dialect="sqlite", database=str(tmp_path / "a.db")), read_only=False).engine_kwargs()

    assert memory["poolclass"] is pool.StaticPool and "pool_size" not in memory
    assert file["pool_size"] == 5 and "poolclass" not in file
    assert memory["hide_parameters"] is True  # error messages never echo bound values


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"database": ":memory:", "host": "h"}, "host is not supported"),
        ({"database": ":memory:", "username": "u"}, "username is not supported"),
        ({"database": ":memory:", "schema": "s"}, "schema is not supported"),
        ({"database": ":memory:", "statement_timeout_seconds": 5}, "statement_timeout_seconds is not supported"),
        ({"database": ":memory:", "isolation_level": "REPEATABLE READ"}, "isolation_level"),
        ({"database": ":memory:", "isolation_level": "AUTOCOMMIT"}, "isolation_level"),
        ({}, "needs modules.database.database"),
    ],
)
def test_sqlite_rejects_options_it_cannot_honor(values: dict[str, Any], message: str) -> None:
    adapter = SqliteAdapter(DatabaseSettings(dialect="sqlite", **values), read_only=False)

    with pytest.raises(DatabaseConfigurationError, match=message):
        adapter.validate()


def test_postgresql_url_and_engine_options() -> None:
    settings = DatabaseSettings(
        dialect="postgresql",
        host="db.example.test",
        port=5433,
        database="app_test",
        username="tester",
        password="hunter2",  # type: ignore[arg-type]
        schema="scratch",
        statement_timeout_seconds=2.5,
        connect_timeout_seconds=3.2,
        isolation_level="repeatable read",
        tls={"mode": "verify-full", "ca_bundle": "ca.pem"},  # type: ignore[arg-type]
    )
    adapter = PostgreSqlAdapter(settings, read_only=True)

    url = adapter.build_url()
    kwargs = adapter.engine_kwargs()

    assert url.drivername == "postgresql+psycopg" and url.host == "db.example.test" and url.port == 5433
    assert url.password == "hunter2" and "hunter2" not in repr(url)
    args = kwargs["connect_args"]
    assert args["connect_timeout"] == 4 and args["sslmode"] == "verify-full" and args["sslrootcert"] == "ca.pem"
    assert args["options"] == "-c default_transaction_read_only=on -c statement_timeout=2500 -c search_path=scratch"
    assert kwargs["isolation_level"] == "REPEATABLE READ" and kwargs["hide_parameters"] is True


def test_postgresql_writable_connections_are_not_forced_read_only() -> None:
    adapter = PostgreSqlAdapter(DatabaseSettings(dialect="postgresql", database="d"), read_only=False)

    assert "options" not in adapter.engine_kwargs()["connect_args"]


def test_postgresql_url_setting_is_normalized_to_psycopg_and_other_drivers_are_rejected() -> None:
    plain = PostgreSqlAdapter(DatabaseSettings(url="postgresql://u:p@h/d"), read_only=False)  # type: ignore[arg-type]
    other = PostgreSqlAdapter(DatabaseSettings(url="postgresql+psycopg2://u:p@h/d"), read_only=False)  # type: ignore[arg-type]

    assert plain.build_url().drivername == "postgresql+psycopg"
    with pytest.raises(DatabaseConfigurationError, match="psycopg"):
        other.validate()


def test_postgresql_needs_a_database_name() -> None:
    with pytest.raises(DatabaseConfigurationError, match="database name"):
        PostgreSqlAdapter(DatabaseSettings(dialect="postgresql", host="h"), read_only=False).validate()


def test_capabilities_document_real_differences() -> None:
    sqlite, postgres = SqliteAdapter.capabilities, PostgreSqlAdapter.capabilities

    assert not sqlite.concurrent_writers and postgres.concurrent_writers
    assert not sqlite.statement_timeout and postgres.statement_timeout
    assert not sqlite.foreign_keys_always_enforced and postgres.foreign_keys_always_enforced
    assert "AUTOCOMMIT" not in sqlite.isolation_levels + postgres.isolation_levels
    with pytest.raises(Exception, match="does not support schemas"):
        SqliteAdapter(DatabaseSettings(dialect="sqlite", database=":memory:"), read_only=False).require("schemas")


def test_adapters_can_be_registered_without_touching_generic_code() -> None:
    class FakeAdapter(SqliteAdapter):
        dialect = "fakedb"

    register_adapter(FakeAdapter)
    register_adapter(FakeAdapter)  # idempotent for the same class

    assert get_adapter_class("fakedb") is FakeAdapter and issubclass(FakeAdapter, DatabaseAdapter)
    with pytest.raises(DatabaseConfigurationError, match="already registered"):
        register_adapter(type("Other", (SqliteAdapter,), {"dialect": "fakedb"}))
    with pytest.raises(DatabaseConfigurationError, match="no adapter for dialect 'oracle'"):
        get_adapter_class("oracle")
