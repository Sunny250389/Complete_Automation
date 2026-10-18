"""Runnable examples of the database module. docs/database_testing.md quotes these tests.

Each example is a test, so it stays correct as the module changes. They run on temporary SQLite databases
except the PostgreSQL one, which skips unless a test database is configured.
"""

from pathlib import Path

import pytest
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, UniqueConstraint, insert

from complete_automation.core.context import RunContext, TestContext, utc_now
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.results import Status, TestResult
from complete_automation.database import (
    DatabaseIntegrityError,
    DatabaseSession,
    EngineManager,
    SchemaValidationError,
    reconcile,
)
from complete_automation.database.example_schema import EXAMPLE_EXPECTATION
from complete_automation.database.reconciliation import ReconciliationConfig, table_rows
from complete_automation.database.reporting import record_connection, record_query, record_transactions
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.validations import (
    ColumnExpectation,
    ForeignKeyExpectation,
    SchemaExpectation,
    TableExpectation,
    assert_no_duplicates,
    assert_record,
    assert_row_count,
    assert_schema,
)

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]

# --- 1. A temporary SQLite database ---------------------------------------------------------------------------


def test_example_1_create_a_temporary_sqlite_database(tmp_path: Path) -> None:
    # sqlite_sandbox_settings grants write/DDL rights on a database this test creates and owns.
    manager = EngineManager(sqlite_sandbox_settings(tmp_path / "shop.db"))

    with manager, manager.session() as db:  # leaving the block rolls back and disposes the engine
        db.execute("CREATE TABLE greetings (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        db.commit()

    assert (tmp_path / "shop.db").is_file()


# --- 2. An example schema ---------------------------------------------------------------------------------------

shop = MetaData()
authors = Table(
    "authors",
    shop,
    Column("id", Integer, primary_key=True),
    Column("email", String(100), nullable=False),
    UniqueConstraint("email", name="uq_authors_email"),
)
books = Table(
    "books",
    shop,
    Column("id", Integer, primary_key=True),
    Column("author_id", Integer, ForeignKey("authors.id"), nullable=False),
    Column("title", String(200), nullable=False),
)


@pytest.fixture
def db(sqlite_database: EngineManager) -> DatabaseSession:
    """A session on a fresh SQLite database that already holds the `shop` schema."""
    with sqlite_database.session() as setup:
        setup.create_all(shop)  # DDL needs allow_ddl, which the sandbox grants
        setup.commit()
    with sqlite_database.session() as session:
        yield session  # type: ignore[misc]


def test_example_2_define_a_schema_with_sqlalchemy_core(db: DatabaseSession) -> None:
    assert set(shop.tables) == {"authors", "books"}
    assert db.query("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").column("name") == ["authors", "books"]


# --- 3. Insert a record safely --------------------------------------------------------------------------------------


def test_example_3_insert_a_record_with_bound_parameters(db: DatabaseSession) -> None:
    hostile = "x'); DROP TABLE authors; --@example.test"  # values are bound, never interpolated

    result = db.execute("INSERT INTO authors (email) VALUES (:email)", {"email": hostile}, expect_rowcount=1)

    assert result.rowcount == 1
    assert_record(db, "authors", {"email": hostile}, {"email": hostile})  # the table still exists
    db.execute_many(insert(authors), [{"email": "a@example.test"}, {"email": "b@example.test"}])  # batch insert


# --- 4. Query with bound parameters ------------------------------------------------------------------------------------


def test_example_4_query_with_named_and_positional_parameters(db: DatabaseSession) -> None:
    db.execute_many(insert(authors), [{"id": 1, "email": "a@example.test"}, {"id": 2, "email": "b@example.test"}])

    named = db.query("SELECT id, email FROM authors WHERE id >= :low ORDER BY id", {"low": 2})
    positional = db.query("SELECT id, email FROM authors WHERE id >= ? ORDER BY id", [2])

    assert named.rows == positional.rows == ({"id": 2, "email": "b@example.test"},)
    assert db.scalar("SELECT count(*) FROM authors") == 2
    assert db.fetch_one("SELECT * FROM authors WHERE id = :id", {"id": 99}) is None


# --- 5. Validate an expected result ---------------------------------------------------------------------------------------


def test_example_5_validate_an_expected_result(db: DatabaseSession) -> None:
    db.execute("INSERT INTO authors (id, email) VALUES (1, 'ada@example.test')")
    db.execute("INSERT INTO books (author_id, title) VALUES (1, 'Notes'), (1, 'Engines')")

    assert_row_count(db, "books", 2, {"author_id": 1})
    assert_record(db, "authors", {"id": 1}, {"email": "ada@example.test"})
    with pytest.raises(AssertionError, match=r"row count of books where author_id=1: expected 3, found 2"):
        assert_row_count(db, "books", 3, {"author_id": 1})


# --- 6. Verify a transaction rollback -----------------------------------------------------------------------------------------


def test_example_6_verify_that_a_rollback_discards_changes(sqlite_database: EngineManager, db: DatabaseSession) -> None:
    db.execute("INSERT INTO authors (email) VALUES ('temp@example.test')")
    assert_row_count(db, "authors", 1)  # visible inside the transaction

    db.rollback()

    assert_row_count(db, "authors", 0)
    with sqlite_database.session() as other:  # a separate connection sees only committed data
        assert_row_count(other, "authors", 0)


# --- 7. Validate schema constraints ---------------------------------------------------------------------------------------------


def test_example_7_validate_schema_and_constraints(db: DatabaseSession) -> None:
    expectation = SchemaExpectation(
        tables=(
            TableExpectation(
                name="authors",
                columns=(ColumnExpectation(name="email", type="string", nullable=False, max_length=100),),
                primary_key=("id",),
                unique_constraints=(("email",),),
            ),
            TableExpectation(
                name="books",
                foreign_keys=(ForeignKeyExpectation(columns=("author_id",), referred_table="authors", referred_columns=("id",)),),
            ),
        )
    )

    assert_schema(db, expectation)  # raises SchemaValidationError listing every difference
    db.execute("INSERT INTO authors (email) VALUES ('once@example.test')")
    with pytest.raises(DatabaseIntegrityError, match="UNIQUE constraint failed: authors.email"):  # the constraint is live
        db.execute("INSERT INTO authors (email) VALUES ('once@example.test')")
    wrong = SchemaExpectation(tables=(TableExpectation(name="authors", columns=(ColumnExpectation(name="email", type="integer"),)),))
    with pytest.raises(SchemaValidationError, match=r"\[type_mismatch\] authors.email: expected integer, actual string"):
        assert_schema(db, wrong)


def test_example_7b_the_bundled_example_schema_and_expectation(example_session: DatabaseSession) -> None:
    assert_schema(example_session, EXAMPLE_EXPECTATION)


# --- 8. Detect duplicate records --------------------------------------------------------------------------------------------------


def test_example_8_detect_duplicate_records(db: DatabaseSession) -> None:
    db.execute("CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, email TEXT, plan TEXT)")  # no unique constraint
    db.execute("INSERT INTO subscriptions (email, plan) VALUES ('a@example.test', 'pro'), ('a@example.test', 'pro'), ('b@example.test', 'pro')")

    with pytest.raises(AssertionError, match=r"duplicates in subscriptions\(email, plan\): 1 duplicated value group"):
        assert_no_duplicates(db, "subscriptions", ["email", "plan"])
    assert_no_duplicates(db, "subscriptions", ["email", "plan"], where={"email": "b@example.test"})
    assert_no_duplicates(db, "authors", "email")  # the constrained table is clean


# --- 9. Compare expected and actual datasets ---------------------------------------------------------------------------------------


def test_example_9_compare_expected_and_actual_datasets(db: DatabaseSession) -> None:
    db.execute_many(insert(authors), [{"id": 1, "email": "a@example.test"}, {"id": 2, "email": "CHANGED@example.test"}])
    expected = [{"id": 1, "email": "a@example.test"}, {"id": 2, "email": "b@example.test"}, {"id": 3, "email": "c@example.test"}]

    result = reconcile(expected, table_rows(db, "SELECT id, email FROM authors"), ReconciliationConfig(key=("id",)))

    assert not result.passed
    assert (result.matched, result.missing_count, result.unexpected_count, result.mismatched_count) == (1, 1, 0, 1)
    assert result.missing == [(3,)] and result.mismatches[0].fields[0].field == "email"
    with pytest.raises(AssertionError, match="reconciliation FAILED"):
        result.assert_matches()  # the message lists missing keys and field-level differences


# --- 10. An optional PostgreSQL test ------------------------------------------------------------------------------------------------


@pytest.mark.postgresql
def test_example_10_optional_postgresql_test(postgres_session: DatabaseSession) -> None:
    # Skips unless run with:  uv run pytest -m postgresql --cauto-env postgres_test
    # (plus CAUTO_MODULES__DATABASE__USERNAME / __PASSWORD in the environment). The fixture created a
    # scratch schema with the example tables and drops it afterwards.
    assert_schema(postgres_session, EXAMPLE_EXPECTATION)
    assert postgres_session.manager.dialect == "postgresql"


# --- 11. Publish a database test result through the shared core ----------------------------------------------------------------------------


def test_example_11_publish_a_result_through_the_shared_core_contracts(
    example_database: EngineManager, result_recorder: ResultRecorder, run_context: RunContext, test_context: TestContext
) -> None:
    started = utc_now()
    summaries: list[dict[str, object]] = []

    with example_database.session(on_query=summaries.append) as db:
        assert_row_count(db, "customers", 3)
        db.rollback()

    # Statements, the connection and the transaction outcome go into the shared ResultRecorder...
    record_connection(result_recorder, example_database)
    for summary in summaries:
        record_query(result_recorder, summary)  # type: ignore[arg-type]
    record_transactions(result_recorder, db.transaction_outcomes)

    # ...and become the one TestResult every module reports. (Under --cauto-report the core plugin builds
    # this for you from the recorder; it is constructed by hand here to show the contract.)
    result = TestResult(
        run_id=run_context.run_id,
        test_id=test_context.test_id,
        name=test_context.name,
        module="database",
        status=Status.PASSED,
        started_at=started,
        finished_at=utc_now(),
        correlation_id=test_context.correlation_id,
        metrics=tuple(result_recorder.metrics),
        metadata=dict(result_recorder.metadata),
    )

    assert TestResult.from_dict(result.to_dict()) == result  # JSON round trip through the shared schema
    assert result.metadata["db_connections"] == [
        {"dialect": "sqlite", "database": "test.db", "environment": run_context.environment, "read_only": False}
    ]
    assert result.metrics[0].name == "db.query_ms" and result.metadata["db_queries"][0]["verb"] == "SELECT"  # type: ignore[index]
    assert result.metadata["db_transactions"][0]["outcome"] == "rolled_back"  # type: ignore[index]


# --- 12. Update, verify, roll back (the short example in docs/database_testing.md) --------------------------------------------------


def test_example_12_update_verify_and_roll_back(example_session: DatabaseSession) -> None:
    db = example_session
    db.execute("UPDATE orders SET status = :s WHERE id = :id", {"s": "paid", "id": 1}, expect_rowcount=1)
    assert_record(db, "orders", {"id": 1}, {"status": "paid"})
    db.rollback()
    assert_record(db, "orders", {"id": 1}, {"status": "new"})
