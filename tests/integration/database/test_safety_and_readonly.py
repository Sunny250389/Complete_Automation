"""Read-only mode, destructive-operation guards and credential handling, on real SQLite databases.

The application-level policy is a guardrail. Where the engine can enforce
read-only itself (SQLite ``mode=ro`` / ``query_only``), these tests also prove
the *database* refuses writes when the policy is bypassed.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy import exc as sa_exc

from complete_automation.database import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseSettings,
    EngineManager,
    UnsafeOperationError,
)
from complete_automation.database.example_schema import orders
from complete_automation.database.validations import count_rows, safe_identifier

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]


def _database_file(example_database: EngineManager) -> Path:
    return Path(example_database.settings.database or "")


def _manager(path: Path, **settings: object) -> EngineManager:
    return EngineManager(DatabaseSettings(dialect="sqlite", database=str(path), **settings))  # type: ignore[arg-type]


# --- read-only mode ---------------------------------------------------------------------------------------


def test_the_default_is_read_only_and_reads_work(example_database: EngineManager) -> None:
    with _manager(_database_file(example_database)) as ro, ro.session() as db:
        assert ro.policy.read_only and count_rows(db, "customers") == 3


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers (email, name) VALUES ('w@example.test', 'W')",
        "UPDATE customers SET name = 'x' WHERE id = 1",
        "DELETE FROM customers WHERE id = 1",
        "CREATE TABLE extra (a INTEGER)",
        "DROP TABLE customers",
        "PRAGMA foreign_keys = OFF",
    ],
)
def test_writes_are_refused_in_read_only_mode_with_an_explanation(example_database: EngineManager, sql: str) -> None:
    with _manager(_database_file(example_database)) as ro, ro.session() as db:
        with pytest.raises(UnsafeOperationError, match="access is read_only"):
            db.execute(sql)
        assert count_rows(db, "customers") == 3


def test_the_database_itself_refuses_writes_when_the_policy_is_bypassed(example_database: EngineManager) -> None:
    with _manager(_database_file(example_database)) as ro, ro.session() as db:
        with pytest.raises(sa_exc.OperationalError, match="readonly"):
            db.connection.exec_driver_sql("INSERT INTO customers (email, name) VALUES ('raw@example.test', 'Raw')")
        db.rollback()
        assert count_rows(db, "customers") == 3


def test_a_read_write_manager_over_the_same_file_still_works(example_database: EngineManager) -> None:
    with _manager(_database_file(example_database), access="read_write") as rw, rw.session() as db:
        assert db.execute("UPDATE customers SET name = 'renamed' WHERE id = 1", expect_rowcount=1).rowcount == 1


def test_read_only_mode_never_creates_a_missing_database(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.db"

    with _manager(missing) as ro:
        health = ro.health_check()
        with pytest.raises(DatabaseConnectionError):
            ro.session()

    assert not health.ok and health.error_kind == "connection" and not missing.exists()


def test_a_read_only_memory_database_is_query_only() -> None:
    with EngineManager(DatabaseSettings(dialect="sqlite", database=":memory:")) as ro, ro.session() as db:
        assert db.scalar("SELECT 1") == 1
        with pytest.raises(sa_exc.OperationalError, match="readonly"):
            db.connection.exec_driver_sql("CREATE TABLE t (a INTEGER)")


def test_protected_environments_force_read_only_and_reject_write_configuration(example_database: EngineManager) -> None:
    path = _database_file(example_database)

    with EngineManager(DatabaseSettings(dialect="sqlite", database=str(path)), environment="prod") as prod, prod.session() as db:
        with pytest.raises(UnsafeOperationError, match="access is read_only|protected"):
            db.execute("UPDATE customers SET name = 'x' WHERE id = 1")
    with pytest.raises(DatabaseConfigurationError, match="protected"):
        EngineManager(DatabaseSettings(dialect="sqlite", database=str(path), access="read_write"), environment="production")


# --- destructive and DDL guards --------------------------------------------------------------------------------------


def test_bulk_delete_needs_destructive_authorization_even_with_write_access(example_database: EngineManager) -> None:
    with _manager(_database_file(example_database), access="read_write") as rw, rw.session() as db:
        with pytest.raises(UnsafeOperationError, match="DELETE without WHERE.*destructive|destructive.*DELETE without WHERE"):
            db.execute("DELETE FROM order_items")
        with pytest.raises(UnsafeOperationError, match="destructive"):
            db.execute(delete(orders))
        assert count_rows(db, "order_items") == 15


def test_ddl_needs_allow_ddl_and_an_allowlisted_database(example_database: EngineManager) -> None:
    path = _database_file(example_database)
    with _manager(path, access="read_write") as rw, rw.session() as db, pytest.raises(UnsafeOperationError, match="allow_ddl"):
        db.execute("CREATE TABLE extra (a INTEGER)")

    with pytest.raises(ValidationError, match="privileged_databases"):
        _manager(path, access="read_write", allow_ddl=True)


def test_drop_is_refused_when_the_database_is_not_on_the_allowlist(example_database: EngineManager) -> None:
    path = _database_file(example_database)

    with _manager(path, access="read_write", allow_destructive=True, privileged_databases=("some_other.db",)) as rw, rw.session() as db:
        with pytest.raises(UnsafeOperationError, match="not in privileged_databases"):
            db.execute("DROP TABLE order_items")
        assert count_rows(db, "order_items") == 15


def test_drop_works_only_on_the_explicitly_allowlisted_database(example_database: EngineManager) -> None:
    path = _database_file(example_database)

    with _manager(path, access="read_write", allow_destructive=True, privileged_databases=(path.name,)) as rw, rw.session() as db:
        db.execute("DROP TABLE order_items")
        db.commit()
        assert "order_items" not in db.query("SELECT name FROM sqlite_master").column("name")


def test_transaction_control_and_multiple_statements_are_refused_as_raw_sql(example_session) -> None:  # type: ignore[no-untyped-def]
    for sql in ("COMMIT", "ROLLBACK", "BEGIN", "SAVEPOINT x", "SET search_path = other"):
        with pytest.raises(UnsafeOperationError, match="transaction API"):
            example_session.execute(sql)
    with pytest.raises(UnsafeOperationError, match="2 statements"):
        example_session.execute("UPDATE customers SET name = 'a' WHERE id = 1; DELETE FROM customers WHERE id = 2")
    assert count_rows(example_session, "customers") == 3


def test_a_read_method_refuses_a_write_statement(example_session) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(Exception, match="use execute"):
        example_session.query("DELETE FROM customers WHERE id = 1")
    with pytest.raises(Exception, match="use execute"):
        example_session.iterate("UPDATE customers SET name = 'x' WHERE id = 1").__next__()


def test_statements_inside_literals_do_not_trip_the_policy(example_session) -> None:  # type: ignore[no-untyped-def]
    note = "please DROP TABLE customers; then DELETE FROM orders"

    example_session.execute("UPDATE customers SET name = :n WHERE id = 1", {"n": note})

    assert example_session.scalar("SELECT name FROM customers WHERE id = 1") == note


# --- identifiers ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["customers; DROP TABLE customers", "a b", "1abc", "t-1", '"quoted"', "", "x" * 80, "a.b"])
def test_dynamic_identifiers_must_be_plain_names(name: str) -> None:
    with pytest.raises(UnsafeOperationError, match="not a plain identifier"):
        safe_identifier(name)


def test_identifier_allowlists_are_honored() -> None:
    assert safe_identifier("customers", allowed=("customers", "orders")) == "customers"
    with pytest.raises(UnsafeOperationError, match="not in the allowed names"):
        safe_identifier("payments", allowed=("customers", "orders"))


def test_validation_helpers_refuse_injected_table_and_column_names(example_session) -> None:  # type: ignore[no-untyped-def]
    from complete_automation.database.validations import assert_row_exists

    with pytest.raises(UnsafeOperationError):
        count_rows(example_session, "customers; DROP TABLE customers")
    with pytest.raises(UnsafeOperationError):
        assert_row_exists(example_session, "customers", {"id = 1 OR 1": 1})
    assert count_rows(example_session, "customers") == 3


# --- credentials ----------------------------------------------------------------------------------------------------------


def test_connection_urls_are_redacted_everywhere_they_surface() -> None:
    settings = DatabaseSettings(url="postgresql+psycopg://tester:hunter2@db.example.test:5432/app_test")  # type: ignore[arg-type]
    pytest.importorskip("psycopg")

    manager = EngineManager(settings)

    for text in (manager.redacted_url, repr(manager), str(manager.settings), repr(manager.settings)):
        assert "hunter2" not in text
    assert "db.example.test" in manager.redacted_url and manager.dialect == "postgresql"


def test_connection_failures_do_not_leak_the_password() -> None:
    pytest.importorskip("psycopg")
    settings = DatabaseSettings(
        dialect="postgresql", host="127.0.0.1", port=1, database="x", username="u", password="hunter2", connect_timeout_seconds=1  # type: ignore[arg-type]
    )

    with EngineManager(settings) as manager:
        health = manager.health_check()
        with pytest.raises(DatabaseConnectionError) as error:
            manager.session()

    assert not health.ok and health.error_kind == "connection"
    for text in (health.error or "", str(error.value), repr(error.value.original)):
        assert "hunter2" not in text
    assert manager.open_connections == 0


def test_constructing_a_manager_never_opens_or_creates_the_file(tmp_path: Path) -> None:
    target = tmp_path / "lazy.db"

    manager = _manager(target)

    assert not target.exists() and manager.open_connections == 0
