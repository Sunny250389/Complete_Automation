"""Schema inspection and validation against the example schema, drift detection and migration checks."""

import pytest

from complete_automation.database import DatabaseSession, DataAssertionError, EngineManager, SchemaValidationError
from complete_automation.database.example_schema import EXAMPLE_EXPECTATION, EXAMPLE_TABLES
from complete_automation.database.validations import (
    ColumnExpectation,
    SchemaExpectation,
    TableExpectation,
    assert_counts_preserved,
    assert_schema,
    assert_schema_changes,
    assert_schema_version,
    assert_table_exists,
    assert_view_exists,
    read_schema_version,
    snapshot_schema,
    table_counts,
    validate_schema,
)

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]


def test_the_example_schema_satisfies_its_expectation(example_session: DatabaseSession) -> None:
    assert validate_schema(example_session, EXAMPLE_EXPECTATION) == []
    assert_schema(example_session, EXAMPLE_EXPECTATION)


def test_validation_also_works_on_a_raw_connection(example_session: DatabaseSession) -> None:
    assert_schema(example_session.connection, EXAMPLE_EXPECTATION)


def test_snapshot_reflects_columns_keys_constraints_and_indexes(example_session: DatabaseSession) -> None:
    snapshot = snapshot_schema(example_session)
    orders = snapshot.table("orders")
    assert orders is not None

    assert {t.name for t in snapshot.tables} == set(EXAMPLE_TABLES) and snapshot.exact_tables
    assert [c.name for c in orders.columns] == ["id", "customer_id", "total", "status", "placed_at"]
    assert orders.primary_key == ("id",)
    assert (orders.foreign_keys[0].referred_table, orders.foreign_keys[0].on_delete) == ("customers", "RESTRICT")
    assert any(ix.name == "ix_orders_customer_id" for ix in orders.indexes)
    customers = snapshot.table("customers")
    assert customers is not None and ("email",) in customers.unique_constraints
    assert {c.name: c.nullable for c in customers.columns}["email"] is False
    assert {c.name: c.max_length for c in customers.columns}["email"] == 255


def test_table_and_view_existence(example_session: DatabaseSession) -> None:
    example_session.execute("CREATE VIEW customer_orders AS SELECT c.id AS customer_id, o.id AS order_id FROM customers c JOIN orders o ON o.customer_id = c.id")

    assert_table_exists(example_session, "orders")
    assert_view_exists(example_session, "customer_orders")
    assert_schema(example_session, SchemaExpectation(views=("customer_orders",)))
    with pytest.raises(SchemaValidationError, match="table 'ghost' does not exist"):
        assert_table_exists(example_session, "ghost")
    with pytest.raises(SchemaValidationError, match="view 'orders' does not exist"):
        assert_view_exists(example_session, "orders")  # a table is not a view


def test_every_difference_is_reported_at_once(example_session: DatabaseSession) -> None:
    example_session.execute("ALTER TABLE customers DROP COLUMN name")
    example_session.execute("DROP INDEX ix_orders_customer_id")
    expectation = EXAMPLE_EXPECTATION.model_copy(
        update={"tables": (*EXAMPLE_EXPECTATION.tables, TableExpectation(name="payments")), "views": ("v_missing",)}
    )

    with pytest.raises(SchemaValidationError) as error:
        assert_schema(example_session, expectation)

    message = str(error.value)
    for expected in (
        "[missing_column] customers.name",
        "[missing_index] orders(customer_id)",
        "[missing_table] payments",
        "[missing_view] v_missing",
    ):
        assert expected in message
    assert "4 difference(s)" in message


def test_a_type_or_nullability_drift_is_not_hidden_by_matching_names(example_session: DatabaseSession) -> None:
    example_session.execute("ALTER TABLE orders RENAME COLUMN total TO total_old")
    example_session.execute("ALTER TABLE orders ADD COLUMN total TEXT")

    with pytest.raises(SchemaValidationError) as error:
        assert_schema(example_session, EXAMPLE_EXPECTATION)

    assert "[type_mismatch] orders.total: expected numeric, actual string" in str(error.value)
    assert "[nullable_mismatch] orders.total: expected False, actual True" in str(error.value)


def test_unique_constraint_and_foreign_key_actions_are_checked(example_session: DatabaseSession) -> None:
    expectation = SchemaExpectation(
        tables=(
            TableExpectation(name="customers", unique_constraints=(("name",),)),
            TableExpectation(
                name="order_items",
                foreign_keys=(
                    EXAMPLE_EXPECTATION.table("order_items").foreign_keys[0].model_copy(update={"on_delete": "SET NULL"}),  # type: ignore[union-attr]
                ),
            ),
        )
    )

    with pytest.raises(SchemaValidationError) as error:
        assert_schema(example_session, expectation)

    assert "[missing_unique_constraint] customers(name)" in str(error.value)
    assert "[foreign_key_on_delete_mismatch]" in str(error.value) and "expected SET NULL, actual CASCADE" in str(error.value)


def test_exact_expectations_flag_extra_columns_and_tables(example_session: DatabaseSession) -> None:
    example_session.execute("CREATE TABLE stray (id INTEGER PRIMARY KEY)")
    example_session.execute("ALTER TABLE customers ADD COLUMN nickname TEXT")
    expectation = EXAMPLE_EXPECTATION.model_copy(
        update={"exact_tables": True, "tables": (EXAMPLE_EXPECTATION.tables[0].model_copy(update={"exact_columns": True}), *EXAMPLE_EXPECTATION.tables[1:])}
    )

    with pytest.raises(SchemaValidationError) as error:
        assert_schema(example_session, expectation)

    assert "[unexpected_table] stray" in str(error.value) and "[unexpected_column] customers.nickname" in str(error.value)


def test_validation_only_reads_the_database(example_database: EngineManager) -> None:
    with example_database.session() as db:
        before = snapshot_schema(db)
        validate_schema(db, SchemaExpectation(tables=(TableExpectation(name="payments", columns=(ColumnExpectation(name="id"),)),)))
        assert snapshot_schema(db) == before  # a failed validation repairs nothing


# --- migrations ----------------------------------------------------------------------------------------------------


def test_a_migration_is_validated_by_before_and_after_snapshots(example_session: DatabaseSession) -> None:
    before = snapshot_schema(example_session)
    counts_before = table_counts(example_session, EXAMPLE_TABLES)

    example_session.execute("ALTER TABLE customers ADD COLUMN phone TEXT")
    example_session.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, note TEXT)")
    after = snapshot_schema(example_session)

    changes = assert_schema_changes(before, after, added_tables=["audit_log"], added_columns={"customers": ["phone"]})
    assert changes.added_columns == {"customers": ("phone",)}
    assert_counts_preserved(counts_before, table_counts(example_session, EXAMPLE_TABLES))
    with pytest.raises(SchemaValidationError, match="added_tables"):
        assert_schema_changes(before, after, added_columns={"customers": ["phone"]})  # audit_log was not declared


def test_a_migration_that_loses_rows_is_caught(example_session: DatabaseSession) -> None:
    counts_before = table_counts(example_session, EXAMPLE_TABLES)

    example_session.execute("DELETE FROM order_items WHERE id <= 3")

    with pytest.raises(DataAssertionError, match="order_items: expected 15, found 12"):
        assert_counts_preserved(counts_before, table_counts(example_session, EXAMPLE_TABLES))
    assert_counts_preserved(counts_before, table_counts(example_session, EXAMPLE_TABLES), allowed_changes={"order_items": -3})


def test_the_recorded_schema_version_is_read_and_checked(example_session: DatabaseSession) -> None:
    example_session.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
    assert read_schema_version(example_session) is None

    example_session.execute("INSERT INTO alembic_version VALUES ('1a2b3c')")

    assert read_schema_version(example_session) == "1a2b3c"
    assert_schema_version(example_session, "1a2b3c")
    with pytest.raises(DataAssertionError, match="expected '9z', found '1a2b3c'"):
        assert_schema_version(example_session, "9z")
    example_session.execute("INSERT INTO alembic_version VALUES ('second')")
    with pytest.raises(DataAssertionError, match="more than one version row"):
        read_schema_version(example_session)
