"""Constraints and referential integrity, verified against the real engine instead of assumed."""

import pytest

from complete_automation.database import DatabaseIntegrityError, DatabaseSession, DataAssertionError, EngineManager
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.validations import (
    Dependent,
    assert_aggregate_matches,
    assert_delete_cascades,
    assert_delete_restricted,
    assert_derived_consistent,
    assert_foreign_keys_enforced,
    assert_no_orphans,
    count_rows,
    find_derived_mismatches,
    find_orphans,
)

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]

ITEMS = [Dependent("order_items", "order_id", "id")]
CENTS = 0.005  # SQLite stores NUMERIC as floating point, so sums need a tolerance


def _derived(db: DatabaseSession, **overrides: object) -> None:
    assert_derived_consistent(
        db,
        parent_table="orders", parent_key="id", parent_column="total",
        child_table="order_items", child_key="order_id", child_column="amount",
        tolerance=CENTS, **overrides,
    )  # fmt: skip


# --- constraints --------------------------------------------------------------------------------------------


def test_foreign_keys_are_enforced_on_the_connection(example_session: DatabaseSession) -> None:
    assert_foreign_keys_enforced(example_session)
    with pytest.raises(DatabaseIntegrityError, match="FOREIGN KEY constraint failed"):
        example_session.execute("INSERT INTO orders (customer_id, total, placed_at) VALUES (999, 1, '2024-01-01')")


def test_a_connection_without_foreign_key_enforcement_is_detected() -> None:
    settings = sqlite_sandbox_settings(":memory:", sqlite={"foreign_keys": False})
    with EngineManager(settings) as manager, manager.session() as db:
        with pytest.raises(DataAssertionError, match="foreign keys are not enforced"):
            assert_foreign_keys_enforced(db)


def test_check_and_not_null_constraints(example_session: DatabaseSession) -> None:
    with pytest.raises(DatabaseIntegrityError, match="CHECK constraint failed"):
        example_session.execute("INSERT INTO order_items (order_id, sku, quantity, amount) VALUES (1, 'X', 0, 1)")
    with pytest.raises(DatabaseIntegrityError, match="NOT NULL constraint failed: orders.total"):
        example_session.execute("INSERT INTO orders (customer_id, total, placed_at) VALUES (1, NULL, '2024-01-01')")


def test_composite_unique_constraint(example_session: DatabaseSession) -> None:
    sku = example_session.scalar("SELECT sku FROM order_items WHERE order_id = 1 LIMIT 1")

    with pytest.raises(DatabaseIntegrityError, match="UNIQUE constraint failed: order_items.order_id, order_items.sku"):
        example_session.execute("INSERT INTO order_items (order_id, sku, quantity, amount) VALUES (1, :s, 1, 1)", {"s": sku})


# --- referential behavior -------------------------------------------------------------------------------------------


def test_restrict_blocks_deleting_a_parent_that_has_children(example_session: DatabaseSession) -> None:
    assert_delete_restricted(example_session, "customers", {"id": 1})

    assert count_rows(example_session, "customers") == 3  # the attempt was rolled back


def test_restrict_check_fails_when_the_delete_is_allowed(example_session: DatabaseSession) -> None:
    example_session.execute("INSERT INTO customers (email, name) VALUES ('lonely@example.test', 'No orders')")

    with pytest.raises(DataAssertionError, match="expected to be restricted by a foreign key, but it succeeded"):
        assert_delete_restricted(example_session, "customers", {"email": "lonely@example.test"})
    assert count_rows(example_session, "customers", {"email": "lonely@example.test"}) == 1  # still undone


def test_cascade_removes_exactly_the_dependents_and_is_rolled_back(example_session: DatabaseSession) -> None:
    before = count_rows(example_session, "order_items")

    assert_delete_cascades(example_session, "orders", {"id": 1}, ITEMS)

    assert count_rows(example_session, "order_items") == before and count_rows(example_session, "orders") == 6


def test_cascade_check_catches_a_missing_cascade(example_session: DatabaseSession) -> None:
    # customers -> orders is RESTRICT, so deleting a customer cannot cascade; the delete itself is rejected
    with pytest.raises(DatabaseIntegrityError):
        assert_delete_cascades(example_session, "customers", {"id": 1}, [Dependent("orders", "customer_id", "id")])
    assert count_rows(example_session, "customers") == 3


def test_cascade_check_needs_a_condition_and_a_matching_row(example_session: DatabaseSession) -> None:
    with pytest.raises(DataAssertionError, match="needs a condition"):
        assert_delete_cascades(example_session, "orders", {}, ITEMS)
    with pytest.raises(DataAssertionError, match="no orders row matched where id=99"):
        assert_delete_cascades(example_session, "orders", {"id": 99}, ITEMS)
    with pytest.raises(DataAssertionError, match="needs a condition"):
        assert_delete_restricted(example_session, "customers", {})


# --- orphans ------------------------------------------------------------------------------------------------------------


def test_orphan_detection_finds_children_without_parents(example_database: EngineManager) -> None:
    settings = example_database.settings.model_copy(update={"sqlite": example_database.settings.sqlite.model_copy(update={"foreign_keys": False})})
    with EngineManager(settings) as lax, lax.session() as db:
        kwargs = {"child_table": "orders", "child_column": "customer_id", "parent_table": "customers", "parent_column": "id"}
        assert_no_orphans(db, **kwargs)  # type: ignore[arg-type]

        db.execute("INSERT INTO orders (customer_id, total, placed_at) VALUES (777, 5, '2024-02-02')")

        report = find_orphans(db, **kwargs)  # type: ignore[arg-type]
        assert report.count == 1 and report.sample == (777,) and report.child == "orders.customer_id"
        with pytest.raises(DataAssertionError, match=r"orphaned rows: 1 row\(s\) in orders.customer_id have no matching customers.id") as hidden:
            assert_no_orphans(db, **kwargs)  # type: ignore[arg-type]
        assert "777" not in str(hidden.value)
        with pytest.raises(DataAssertionError, match="777"):
            assert_no_orphans(db, show_values=True, **kwargs)  # type: ignore[arg-type]


def test_null_foreign_keys_are_not_orphans(example_database: EngineManager) -> None:
    with example_database.session() as db:
        db.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, order_id INTEGER)")
        db.execute("INSERT INTO tickets (order_id) VALUES (NULL), (1)")

        assert_no_orphans(db, child_table="tickets", child_column="order_id", parent_table="orders", parent_column="id")


# --- derived fields and source/target comparison --------------------------------------------------------------------------------


def test_stored_totals_match_the_sum_of_their_items(example_session: DatabaseSession) -> None:
    _derived(example_session)
    assert find_derived_mismatches(
        example_session, parent_table="orders", parent_key="id", parent_column="total",
        child_table="order_items", child_key="order_id", child_column="amount", tolerance=CENTS,
    ) == []  # fmt: skip


def test_a_wrong_stored_total_is_reported_with_both_values(example_session: DatabaseSession) -> None:
    example_session.execute("UPDATE orders SET total = 1.00 WHERE id = 2")

    with pytest.raises(DataAssertionError, match=r"derived field orders.total is inconsistent in 1 row\(s\): id=2: stored 1, computed 97\.47") as error:
        _derived(example_session)
    assert "id=1" not in str(error.value)


def test_orders_without_items_are_computed_as_zero(example_session: DatabaseSession) -> None:
    example_session.execute("INSERT INTO orders (customer_id, total, placed_at) VALUES (1, 0, '2024-03-03')")
    _derived(example_session)

    example_session.execute("UPDATE orders SET total = 9 WHERE id = 7")
    with pytest.raises(DataAssertionError, match="id=7: stored 9, computed 0"):
        _derived(example_session)


def test_count_based_derived_fields(example_session: DatabaseSession) -> None:
    example_session.execute("CREATE TABLE tallies (order_id INTEGER PRIMARY KEY, item_count INTEGER)")
    example_session.execute("INSERT INTO tallies SELECT order_id, count(*) FROM order_items GROUP BY order_id")
    example_session.execute("INSERT INTO tallies VALUES (7, 1)")

    mismatches = find_derived_mismatches(
        example_session, parent_table="tallies", parent_key="order_id", parent_column="item_count",
        child_table="order_items", child_key="order_id", function="count",
    )  # fmt: skip

    assert [(m.key, m.stored, m.computed) for m in mismatches] == [(7, 1, 0)]


def test_aggregates_can_be_compared_across_tables(example_session: DatabaseSession) -> None:
    example_session.execute("CREATE TABLE order_archive (id INTEGER PRIMARY KEY, total NUMERIC)")
    example_session.execute("INSERT INTO order_archive SELECT id, total FROM orders")

    assert_aggregate_matches(example_session, source_table="orders", target_table="order_archive")
    assert_aggregate_matches(example_session, source_table="orders", target_table="order_archive", function="sum", source_column="total", tolerance=CENTS)

    example_session.execute("DELETE FROM order_archive WHERE id = 6")
    with pytest.raises(DataAssertionError, match=r"count\(\*\) differs: orders = 6, order_archive = 5"):
        assert_aggregate_matches(example_session, source_table="orders", target_table="order_archive")
    assert_aggregate_matches(
        example_session, source_table="orders", target_table="order_archive", source_where={"id": [1, 2]}, target_where={"id": [1, 2]}
    )
