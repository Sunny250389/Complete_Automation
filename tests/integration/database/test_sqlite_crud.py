"""CRUD, parameterized queries and result handling against a real temporary SQLite database."""

import datetime as dt
import traceback
from decimal import Decimal

import pytest
from sqlalchemy import insert, select, update

from complete_automation.database import (
    DatabaseIntegrityError,
    DatabaseSession,
    EngineManager,
    ResultTooLargeError,
)
from complete_automation.database.errors import DataAssertionError
from complete_automation.database.example_schema import customers, orders
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.validations import (
    assert_pagination_consistent,
    assert_record,
    assert_row_absent,
    assert_row_count,
    assert_row_exists,
    assert_sorted,
    count_rows,
)

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]

NEW = {"email": "new.user@example.test", "name": "New User"}


# --- create ------------------------------------------------------------------------------------------


def test_insert_reports_affected_rows_and_defaults_are_applied(example_session: DatabaseSession) -> None:
    result = example_session.execute(
        "INSERT INTO customers (email, name) VALUES (:email, :name)", NEW, expect_rowcount=1
    )

    assert result.rowcount == 1 and result.verb == "INSERT" and result.rows == ()
    row = assert_row_exists(example_session, "customers", {"email": NEW["email"]})
    assert row["status"] == "active" and row["id"] == 4  # generated identifier follows the seed
    assert row["created_at"]  # server-generated timestamp


def test_generated_identifiers_are_retrievable_with_returning(example_session: DatabaseSession) -> None:
    result = example_session.execute(insert(customers).values(**NEW).returning(customers.c.id))

    assert result.scalar() == 4 and result.rowcount in (1, None)


def test_unique_email_is_enforced(example_session: DatabaseSession) -> None:
    existing = example_session.scalar("SELECT email FROM customers WHERE id = 1")

    with pytest.raises(DatabaseIntegrityError, match="UNIQUE constraint failed: customers.email") as error:
        example_session.execute("INSERT INTO customers (email, name) VALUES (:e, 'Dup')", {"e": existing})

    assert error.value.kind == "integrity"
    assert existing not in str(error.value)  # the value stays out of the message
    assert existing not in "".join(traceback.format_exception(error.value))  # and out of the printed traceback


def test_batch_insert_with_execute_many(example_session: DatabaseSession) -> None:
    rows = [{"email": f"batch{n}@example.test", "name": f"Batch {n}"} for n in range(5)]

    result = example_session.execute_many(insert(customers), rows)

    assert result.rowcount == 5 and count_rows(example_session, "customers") == 8


# --- read ------------------------------------------------------------------------------------------------


def test_read_by_primary_key_and_by_several_conditions(example_session: DatabaseSession) -> None:
    by_key = assert_row_exists(example_session, "orders", {"id": 3})
    by_two = assert_row_exists(example_session, "orders", {"customer_id": 2, "id": 3})

    assert by_key == by_two and by_key["customer_id"] == 2
    assert_row_absent(example_session, "orders", {"customer_id": 1, "id": 3})


def test_selected_columns_and_result_helpers(example_session: DatabaseSession) -> None:
    result = example_session.query("SELECT id, name FROM customers ORDER BY id")

    assert result.columns == ("id", "name") and len(result) == 3 and result.row_count == 3
    assert result.column("name") == ["Customer 1", "Customer 2", "Customer 3"]
    assert result.first() == {"id": 1, "name": "Customer 1"} and list(result)[2]["id"] == 3
    assert result.operation.value == "read" and result.rowcount is None and result.duration_ms >= 0
    with pytest.raises(DataAssertionError, match="exactly one row, got 3"):
        result.one()
    with pytest.raises(DataAssertionError, match="no column 'missing'"):
        result.column("missing")


def test_scalars_and_aggregates(example_session: DatabaseSession) -> None:
    assert example_session.scalar("SELECT count(*) FROM order_items") == 15
    assert example_session.scalar("SELECT max(quantity) FROM order_items WHERE order_id = :o", {"o": 1}) >= 1
    with pytest.raises(DataAssertionError, match="got 0"):
        example_session.scalar("SELECT id FROM customers WHERE id = -1")
    assert example_session.fetch_one("SELECT id FROM customers WHERE id = -1") is None


def test_null_and_boundary_values(example_session: DatabaseSession) -> None:
    example_session.execute("INSERT INTO customers (email, name) VALUES ('n@example.test', :n)", {"n": "x" * 100})
    example_session.execute("INSERT INTO customers (email, name, status) VALUES ('e@example.test', 'Edge', 'inactive')")

    assert example_session.scalar("SELECT length(name) FROM customers WHERE email = 'n@example.test'") == 100
    assert_row_count(example_session, "customers", 1, {"status": "inactive"})
    assert_row_count(example_session, "customers", 0, {"status": None})  # IS NULL, not = NULL
    with pytest.raises(DatabaseIntegrityError, match="NOT NULL constraint failed"):
        example_session.execute("INSERT INTO customers (email, name) VALUES ('z@example.test', NULL)")


def test_sorting_and_pagination(example_session: DatabaseSession) -> None:
    ordered = example_session.query("SELECT id, placed_at FROM orders ORDER BY placed_at DESC")
    assert_sorted(ordered, "placed_at", descending=True)
    with pytest.raises(DataAssertionError, match="not sorted ascending"):
        assert_sorted(ordered, "placed_at")

    assert assert_pagination_consistent(example_session, "orders", "id", page_size=4) == 6


def test_streaming_reads_in_chunks_without_loading_everything(example_session: DatabaseSession) -> None:
    streamed = list(example_session.iterate("SELECT id FROM order_items ORDER BY id", chunk_size=4))

    assert [row["id"] for row in streamed] == list(range(1, 16))
    assert example_session.history[-1]["rows"] == 15 and example_session.history[-1]["outcome"] == "ok"


def test_large_results_are_refused_unless_streamed(example_database: EngineManager) -> None:
    settings = example_database.settings.model_copy(update={"max_rows": 5})

    with EngineManager(settings) as small, small.session() as db:
        with pytest.raises(ResultTooLargeError, match="more than 5 rows"):
            db.query("SELECT * FROM order_items")
        assert len(db.query("SELECT * FROM order_items", max_rows=100)) == 15
        assert sum(1 for _ in db.iterate("SELECT * FROM order_items")) == 15


# --- update ------------------------------------------------------------------------------------------------------


def test_update_reports_affected_rows_and_persists_values(example_session: DatabaseSession) -> None:
    result = example_session.execute(
        "UPDATE customers SET status = :s WHERE id IN (1, 2)", {"s": "inactive"}, expect_rowcount=2
    )

    assert result.rowcount == 2
    assert_record(example_session, "customers", {"id": 1}, {"status": "inactive", "name": "Customer 1"})
    assert_record(example_session, "customers", {"id": 3}, {"status": "active"})


def test_an_update_matching_nothing_affects_zero_rows(example_session: DatabaseSession) -> None:
    assert example_session.execute("UPDATE customers SET name = 'x' WHERE id = -1").rowcount == 0
    with pytest.raises(DataAssertionError, match="affected 0 row.*expected 1"):
        example_session.execute("UPDATE customers SET name = 'x' WHERE id = -1", expect_rowcount=1)


def test_invalid_updates_are_rejected_by_constraints(example_session: DatabaseSession) -> None:
    with pytest.raises(DatabaseIntegrityError, match="CHECK constraint failed"):
        example_session.execute("UPDATE customers SET status = 'bogus' WHERE id = 1")
    with pytest.raises(DatabaseIntegrityError, match="UNIQUE"):
        example_session.execute(
            "UPDATE customers SET email = (SELECT email FROM customers WHERE id = 2) WHERE id = 1"
        )


def test_core_expressions_update_with_typed_values(example_session: DatabaseSession) -> None:
    placed = dt.datetime(2030, 5, 17, 8, 30)

    example_session.execute(update(orders).where(orders.c.id == 1).values(total=Decimal("12.50"), placed_at=placed))

    row = example_session.query(select(orders.c.total, orders.c.placed_at).where(orders.c.id == 1)).one()
    assert Decimal(str(row["total"])) == Decimal("12.50") and str(row["placed_at"]).startswith("2030-05-17 08:30")


# --- delete -------------------------------------------------------------------------------------------------------


def test_delete_removes_only_the_matching_rows(example_session: DatabaseSession) -> None:
    example_session.execute("INSERT INTO customers (email, name) VALUES ('gone@example.test', 'Gone')")

    deleted = example_session.execute("DELETE FROM customers WHERE email = :e", {"e": "gone@example.test"}, expect_rowcount=1)

    assert deleted.rowcount == 1
    assert_row_absent(example_session, "customers", {"email": "gone@example.test"})
    assert_row_count(example_session, "customers", 3)


# --- parameters ------------------------------------------------------------------------------------------------------


def test_named_and_positional_parameters_bind_values(example_session: DatabaseSession) -> None:
    named = example_session.query("SELECT id FROM customers WHERE id = :a OR id = :b ORDER BY id", {"a": 1, "b": 3})
    positional = example_session.query("SELECT id FROM customers WHERE id = ? OR id = ? ORDER BY id", [1, 3])

    assert named.column("id") == positional.column("id") == [1, 3]
    with pytest.raises(Exception, match="positional placeholder"):
        example_session.query("SELECT id FROM customers WHERE id = ?", [1, 2])


def test_a_question_mark_inside_a_literal_is_not_a_placeholder(example_session: DatabaseSession) -> None:
    assert example_session.scalar("SELECT '?' || ?", ["!"]) == "?!"


def test_values_that_look_like_sql_are_stored_verbatim(example_session: DatabaseSession) -> None:
    payload = "x'); DROP TABLE customers; --"

    example_session.execute("INSERT INTO customers (email, name) VALUES (:e, :n)", {"e": "inj@example.test", "n": payload})

    assert example_session.scalar("SELECT name FROM customers WHERE email = 'inj@example.test'") == payload
    assert count_rows(example_session, "customers") == 4  # the table is intact


def test_error_messages_never_contain_statement_text_or_bound_values(example_session: DatabaseSession) -> None:
    secret = "alice.private@example.test"

    with pytest.raises(DatabaseIntegrityError) as error:
        example_session.execute("INSERT INTO customers (email, name) VALUES (:e, NULL)", {"e": secret})

    text = "".join(traceback.format_exception(error.value))
    assert secret not in text and "INSERT INTO" not in str(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__  # not chained into the traceback
    assert secret not in str(error.value.original)  # SQLAlchemy hides parameters too
    assert type(error.value.original).__name__ == "IntegrityError"  # the cause is not lost


def test_unrecoverable_sql_mistakes_are_classified_as_syntax_errors(example_session: DatabaseSession) -> None:
    from complete_automation.core.exceptions import ErrorCategory

    with pytest.raises(Exception, match="no such table") as error:
        example_session.query("SELECT * FROM nonexistent_table")

    assert error.value.kind == "syntax" and error.value.category is ErrorCategory.UNKNOWN  # type: ignore[attr-defined]
    assert example_session.history[-1]["outcome"] == "error" and example_session.history[-1]["error_kind"] == "syntax"


def test_a_file_database_can_be_created_from_scratch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manager = EngineManager(sqlite_sandbox_settings(tmp_path / "fresh.db"))

    with manager, manager.session() as db:
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT NOT NULL)")
        db.execute("INSERT INTO t (label) VALUES (?)", ["first"])
        db.commit()

    assert (tmp_path / "fresh.db").is_file() and manager.open_connections == 0
