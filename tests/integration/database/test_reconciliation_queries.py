"""Reconciling real query results, within one database and between two."""

import pytest

from complete_automation.database import (
    DatabaseSession,
    DataAssertionError,
    EngineManager,
    ReconciliationConfig,
    ReconciliationError,
    reconcile,
    reconcile_queries,
)
from complete_automation.database.example_schema import create_example_schema, seed_example_data
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.reconciliation import key_config, table_rows

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]

ORDERS = "SELECT id, customer_id, total, status FROM orders ORDER BY id"
CONFIG = ReconciliationConfig(key=("id",), default_numeric_tolerance=0.005)


@pytest.fixture
def replica(tmp_path) -> EngineManager:  # type: ignore[no-untyped-def]
    """A second, independent database holding the same seed data."""
    manager = EngineManager(sqlite_sandbox_settings(tmp_path / "replica.db"))
    with manager.session() as db:
        create_example_schema(db)
        seed_example_data(db)
        db.commit()
    yield manager  # type: ignore[misc]
    manager.dispose()


def test_two_identical_databases_reconcile(example_database: EngineManager, replica: EngineManager) -> None:
    with example_database.session() as source, replica.session() as target:
        result = reconcile_queries(source, ORDERS, target, ORDERS, CONFIG)

    result.assert_matches()
    assert result.matched == 6 and result.expected_count == result.actual_count == 6


def test_missing_unexpected_and_changed_rows_are_found_across_databases(
    example_database: EngineManager, replica: EngineManager
) -> None:
    with replica.session() as db:
        db.execute("DELETE FROM order_items WHERE order_id = 6")
        db.execute("DELETE FROM orders WHERE id = 6")
        db.execute("UPDATE orders SET status = 'shipped' WHERE id = 2")
        db.execute("INSERT INTO orders (id, customer_id, total, placed_at) VALUES (50, 1, 5, '2024-05-05')")
        db.commit()

    with example_database.session() as source, replica.session() as target:
        result = reconcile_queries(source, ORDERS, target, ORDERS, CONFIG)

    assert (result.missing, result.unexpected, result.mismatched_count) == ([(6,)], [(50,)], 1)
    assert [(f.field, f.expected, f.actual) for f in result.mismatches[0].fields] == [("status", "new", "shipped")]
    with pytest.raises(DataAssertionError, match=r"missing 1, unexpected 1, mismatched 1"):
        result.assert_matches()


def test_reconciling_a_query_against_expected_values_in_memory(example_session: DatabaseSession) -> None:
    expected = [{"id": 1, "name": "Customer 1"}, {"id": 2, "name": "Customer 2"}, {"id": 3, "name": "Customer 3"}]

    result = reconcile(expected, table_rows(example_session, "SELECT id, name FROM customers"), key_config("id"))

    assert result.passed


def test_unordered_queries_are_rejected_instead_of_giving_a_wrong_answer(
    example_database: EngineManager, replica: EngineManager
) -> None:
    descending = "SELECT id, status FROM orders ORDER BY id DESC"

    with example_database.session() as source, replica.session() as target:
        with pytest.raises(ReconciliationError, match="not ordered by the key"):
            reconcile_queries(source, ORDERS, target, descending, CONFIG)


def test_the_two_streams_need_two_connections(example_session: DatabaseSession) -> None:
    with pytest.raises(ReconciliationError, match="two sessions"):
        reconcile_queries(example_session, ORDERS, example_session, ORDERS, CONFIG)


def test_a_large_table_is_compared_in_chunks(example_database: EngineManager, replica: EngineManager) -> None:
    rows = [{"id": n, "n": n % 7} for n in range(1, 3001)]
    for manager in (example_database, replica):
        with manager.session() as db:
            db.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, n INTEGER)")
            db.execute_many("INSERT INTO big (id, n) VALUES (:id, :n)", rows)
            db.commit()
    with replica.session() as db:
        db.execute("UPDATE big SET n = -1 WHERE id = 1500")
        db.commit()

    with example_database.session() as source, replica.session() as target:
        result = reconcile_queries(source, "SELECT id, n FROM big ORDER BY id", target, "SELECT id, n FROM big ORDER BY id", key_config("id"), chunk_size=200)

    assert result.expected_count == 3000 and result.mismatched_count == 1 and result.mismatches[0].key == (1500,)
