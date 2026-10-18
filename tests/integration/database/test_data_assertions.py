"""Data assertions: each passes on good data and fails with a precise, safe message on bad data."""

import datetime as dt
from decimal import Decimal

import pytest

from complete_automation.database import DatabaseSession, DataAssertionError
from complete_automation.database.validations import (
    assert_aggregate,
    assert_distinct_count,
    assert_grouped_counts,
    assert_no_duplicates,
    assert_not_null,
    assert_record,
    assert_recent,
    assert_required_fields,
    assert_row_absent,
    assert_row_count,
    assert_row_count_between,
    assert_row_exists,
    assert_string_length,
    assert_table_empty,
    assert_table_not_empty,
    assert_values_in_range,
    assert_values_in_set,
    find_duplicates,
)

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]


@pytest.fixture
def db(example_session: DatabaseSession) -> DatabaseSession:
    return example_session


# --- counts and existence ---------------------------------------------------------------------------


def test_row_counts(db: DatabaseSession) -> None:
    assert_row_count(db, "customers", 3)
    assert_row_count(db, "orders", 2, {"customer_id": 1})
    assert_row_count_between(db, "order_items", 10, 20)
    assert_table_not_empty(db, "customers")
    db.execute("DELETE FROM order_items WHERE id > 0")
    assert_table_empty(db, "order_items")


def test_row_count_failures_name_the_table_condition_and_both_counts(db: DatabaseSession) -> None:
    with pytest.raises(DataAssertionError, match=r"row count of orders where customer_id=1: expected 5, found 2"):
        assert_row_count(db, "orders", 5, {"customer_id": 1})
    with pytest.raises(DataAssertionError, match=r"expected 0\.\.1, found 6"):
        assert_row_count_between(db, "orders", 0, 1)
    db.execute("DELETE FROM order_items WHERE id > 0")
    db.execute("DELETE FROM orders WHERE id > 0")
    with pytest.raises(DataAssertionError, match="expected at least one row, found none"):
        assert_table_not_empty(db, "orders")


def test_existence_and_absence(db: DatabaseSession) -> None:
    assert assert_row_exists(db, "customers", {"id": 2})["name"] == "Customer 2"
    assert_row_absent(db, "customers", {"id": 99})
    with pytest.raises(DataAssertionError, match="no row found where id=99"):
        assert_row_exists(db, "customers", {"id": 99})
    with pytest.raises(DataAssertionError, match="found more than one"):
        assert_row_exists(db, "customers", {"status": "active"})
    with pytest.raises(DataAssertionError, match="expected no row where id=1, found 1"):
        assert_row_absent(db, "customers", {"id": 1})
    with pytest.raises(DataAssertionError, match="needs at least one condition"):
        assert_row_exists(db, "customers", {})


def test_in_conditions_match_several_values(db: DatabaseSession) -> None:
    assert_row_count(db, "customers", 2, {"id": [1, 3]})


# --- field values ------------------------------------------------------------------------------------------------


def test_record_values_with_exact_and_tolerant_comparison(db: DatabaseSession) -> None:
    db.execute("UPDATE orders SET total = 10.25 WHERE id = 1")

    assert_record(db, "orders", {"id": 1}, {"customer_id": 1, "total": 10.25})
    assert_record(db, "orders", {"id": 1}, {"total": Decimal("10.2501")}, tolerance=Decimal("0.001"))
    with pytest.raises(DataAssertionError, match=r"orders where id=1: 1 field\(s\) differ: total: expected 10.3, found 10.25"):
        assert_record(db, "orders", {"id": 1}, {"total": 10.3})
    with pytest.raises(DataAssertionError, match="nickname: column does not exist"):
        assert_record(db, "orders", {"id": 1}, {"nickname": "x"})


def test_failure_messages_mask_sensitive_columns_and_truncate_long_values(db: DatabaseSession) -> None:
    db.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, password TEXT, bio TEXT)")
    db.execute("INSERT INTO accounts VALUES (1, 'hunter2', :bio)", {"bio": "b" * 500})

    with pytest.raises(DataAssertionError) as error:
        assert_record(db, "accounts", {"id": 1}, {"password": "hunter3", "bio": "c" * 500})

    message = str(error.value)
    assert "hunter2" not in message and "hunter3" not in message and "password: expected ***, found ***" in message
    assert "b" * 100 not in message and "..." in message


def test_required_fields_and_nulls(db: DatabaseSession) -> None:
    assert_required_fields({"a": 1, "b": "x"}, ["a", "b"])
    with pytest.raises(DataAssertionError, match=r"record: required field\(s\) missing or empty: b, c"):
        assert_required_fields({"a": 1, "b": None}, ["a", "b", "c"])
    assert_not_null(db, "orders", ["customer_id", "total"])
    db.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    db.execute("INSERT INTO notes (body) VALUES (NULL), ('x'), (NULL)")
    with pytest.raises(DataAssertionError, match=r"notes: NULL values found in body \(2\)"):
        assert_not_null(db, "notes", "body")


# --- duplicates ---------------------------------------------------------------------------------------------------------


def test_duplicate_detection(db: DatabaseSession) -> None:
    db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, code TEXT, region TEXT)")
    db.execute("INSERT INTO events (code, region) VALUES ('a','eu'), ('a','eu'), ('a','us'), ('b','eu'), (NULL,'eu'), (NULL,'eu')")

    assert_no_duplicates(db, "events", ["code", "region"], {"region": "us"})
    groups = find_duplicates(db, "events", ["code", "region"])
    assert [(g.values, g.count) for g in groups] == [(("a", "eu"), 2)]  # NULLs never count as duplicates
    with pytest.raises(DataAssertionError, match=r"duplicates in events\(code, region\): 1 duplicated value group"):
        assert_no_duplicates(db, "events", ["code", "region"])
    with pytest.raises(DataAssertionError, match=r"\('a', 'eu'\) x2"):
        assert_no_duplicates(db, "events", ["code", "region"], show_values=True)


def test_duplicate_values_are_hidden_unless_requested(db: DatabaseSession) -> None:
    db.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT)")
    db.execute("INSERT INTO people (email) VALUES ('same@example.test'), ('same@example.test')")

    with pytest.raises(DataAssertionError) as hidden:
        assert_no_duplicates(db, "people", "email")
    with pytest.raises(DataAssertionError) as shown:
        assert_no_duplicates(db, "people", "email", show_values=True)

    assert "same@example.test" not in str(hidden.value) and "same@example.test" in str(shown.value)
    assert_no_duplicates(db, "customers", "email")  # the example schema's unique column


# --- value rules ----------------------------------------------------------------------------------------------------------------


def test_numeric_range_and_boundaries(db: DatabaseSession) -> None:
    assert_values_in_range(db, "order_items", "quantity", 1, 4)
    assert_values_in_range(db, "order_items", "quantity", minimum=1)
    assert_values_in_range(db, "order_items", "quantity", maximum=4, where={"order_id": 1})
    with pytest.raises(DataAssertionError, match=r"order_items.quantity: \d+ row\(s\) outside \[2, \+inf\]"):
        assert_values_in_range(db, "order_items", "quantity", minimum=2)
    with pytest.raises(DataAssertionError, match="needs a minimum or a maximum"):
        assert_values_in_range(db, "order_items", "quantity")


def test_date_and_timestamp_ranges(db: DatabaseSession) -> None:
    low, high = dt.datetime(2024, 1, 1), dt.datetime(2024, 12, 31)

    assert_values_in_range(db, "orders", "placed_at", low, high)
    with pytest.raises(DataAssertionError, match="outside"):
        assert_values_in_range(db, "orders", "placed_at", low, dt.datetime(2024, 1, 3))
    assert_recent(db, "orders", "placed_at", within=dt.timedelta(days=10), now=dt.datetime(2024, 1, 10))
    with pytest.raises(DataAssertionError, match="older than"):
        assert_recent(db, "orders", "placed_at", within=dt.timedelta(days=1), now=dt.datetime(2025, 1, 1))
    with pytest.raises(DataAssertionError, match="no values to check"):
        assert_recent(db, "orders", "placed_at", within=dt.timedelta(days=1), now=dt.datetime(2025, 1, 1), where={"id": 99})


def test_string_length_rules(db: DatabaseSession) -> None:
    assert_string_length(db, "customers", "name", min_length=5, max_length=100)
    with pytest.raises(DataAssertionError, match=r"customers.name: 3 row\(s\) with length outside \[0, 5\]"):
        assert_string_length(db, "customers", "name", max_length=5)


def test_allowed_value_sets(db: DatabaseSession) -> None:
    assert_values_in_set(db, "customers", "status", {"active", "inactive"})
    with pytest.raises(DataAssertionError, match=r"1 unexpected distinct value\(s\): 'active'"):
        assert_values_in_set(db, "customers", "status", {"inactive"})
    db.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY, label TEXT)")
    db.execute("INSERT INTO tags (label) VALUES ('a'), (NULL)")
    with pytest.raises(DataAssertionError, match="unexpected distinct"):
        assert_values_in_set(db, "tags", "label", {"a"})
    assert_values_in_set(db, "tags", "label", {"a"}, allow_null=True)


def test_distinct_and_grouped_counts(db: DatabaseSession) -> None:
    assert_distinct_count(db, "orders", "customer_id", 3)
    assert_grouped_counts(db, "orders", "customer_id", {1: 2, 2: 2, 3: 2})
    assert_grouped_counts(db, "orders", "customer_id", {1: 2}, allow_extra_groups=True)
    with pytest.raises(DataAssertionError, match=r"2: expected 5, found 2; 3: unexpected group \(2 rows\)"):
        assert_grouped_counts(db, "orders", "customer_id", {1: 2, 2: 5})
    with pytest.raises(DataAssertionError, match="distinct customer_id.*expected 9, found 3"):
        assert_distinct_count(db, "orders", "customer_id", 9)


def test_aggregates_with_and_without_tolerance(db: DatabaseSession) -> None:
    total = db.scalar("SELECT sum(total) FROM orders")

    assert_aggregate(db, "orders", "sum", "total", total)
    assert_aggregate(db, "orders", "sum", "total", round(total, 1), tolerance=0.1)
    assert_aggregate(db, "orders", "count", None, 6)
    assert_aggregate(db, "orders", "min", "id", 1)
    assert_aggregate(db, "orders", "max", "id", 6, {"customer_id": [1, 2, 3]})
    with pytest.raises(DataAssertionError, match=r"count\(\*\) of orders: expected 7, found 6"):
        assert_aggregate(db, "orders", "count", None, 7)
    with pytest.raises(DataAssertionError, match="unsupported aggregate"):
        assert_aggregate(db, "orders", "median", "id", 1)  # type: ignore[arg-type]
