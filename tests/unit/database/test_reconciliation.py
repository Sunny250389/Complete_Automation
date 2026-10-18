"""Reconciliation logic on plain records: explicit keys, separate missing/unexpected/mismatched reporting."""

import datetime as dt
from decimal import Decimal

import pytest

from complete_automation.core.redaction import Redactor
from complete_automation.database.errors import DataAssertionError, ReconciliationError
from complete_automation.database.reconciliation import (
    ReconciliationConfig,
    compare_records,
    key_config,
    reconcile,
    reconcile_ordered,
)

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]

EXPECTED = [{"id": 1, "name": "a", "amount": Decimal("10.00")}, {"id": 2, "name": "b", "amount": Decimal("20.00")}]


def test_identical_datasets_match() -> None:
    result = reconcile(EXPECTED, [dict(r) for r in EXPECTED], key_config("id"))

    assert result.passed and result.matched == 2 and result.expected_count == result.actual_count == 2
    result.assert_matches()


def test_missing_unexpected_and_mismatched_records_are_reported_separately() -> None:
    actual = [{"id": 1, "name": "CHANGED", "amount": Decimal("10.00")}, {"id": 3, "name": "c", "amount": Decimal("1")}]

    result = reconcile(EXPECTED, actual, key_config("id"))

    assert (result.missing_count, result.unexpected_count, result.mismatched_count, result.matched) == (1, 1, 1, 0)
    assert result.missing == [(2,)] and result.unexpected == [(3,)]
    assert [(f.field, f.expected, f.actual) for f in result.mismatches[0].fields] == [("name", "a", "CHANGED")]
    with pytest.raises(DataAssertionError, match=r"missing 1, unexpected 1, mismatched 1"):
        result.assert_matches()


def test_the_key_is_explicit_and_may_be_composite() -> None:
    with pytest.raises(ReconciliationError, match="explicit key"):
        ReconciliationConfig(key=())
    with pytest.raises(ReconciliationError, match="duplicate"):
        ReconciliationConfig(key=("id", "id"))
    rows_a = [{"a": 1, "b": 1, "v": "x"}, {"a": 1, "b": 2, "v": "y"}]

    assert reconcile(rows_a, list(reversed(rows_a)), key_config("a", "b")).passed
    with pytest.raises(ReconciliationError, match="missing key column 'b'"):
        reconcile([{"a": 1}], [], key_config("a", "b"))


def test_ignored_and_selected_fields() -> None:
    expected = [{"id": 1, "name": "a", "updated_at": "monday"}]
    actual = [{"id": 1, "name": "a", "updated_at": "tuesday"}]

    assert not reconcile(expected, actual, key_config("id")).passed
    assert reconcile(expected, actual, key_config("id", ignore_fields=frozenset({"updated_at"}))).passed
    assert reconcile(expected, actual, key_config("id", compare_fields=("name",))).passed


def test_fields_present_on_only_one_side_are_mismatches() -> None:
    diffs = compare_records({"id": 1, "a": 1}, {"id": 1, "b": 2}, key_config("id"))

    assert {(d.field, d.expected, d.actual) for d in diffs} == {("a", 1, "<absent>"), ("b", "<absent>", 2)}


def test_numeric_tolerance_is_explicit_per_field_or_default() -> None:
    expected, actual = [{"id": 1, "x": 10.0, "y": 10.0}], [{"id": 1, "x": 10.004, "y": 10.5}]

    assert not reconcile(expected, actual, key_config("id")).passed
    strict_y = key_config("id", tolerances={"x": 0.01}, default_numeric_tolerance=0)
    result = reconcile(expected, actual, strict_y)
    assert [f.field for f in result.mismatches[0].fields] == ["y"]
    assert reconcile(expected, actual, key_config("id", default_numeric_tolerance=1)).passed


def test_decimal_and_float_compare_by_value_and_tolerance_never_applies_to_booleans() -> None:
    assert reconcile([{"id": 1, "v": Decimal("1.10")}], [{"id": 1, "v": 1.1}], key_config("id")).passed
    assert reconcile([{"id": 1, "v": True}], [{"id": 1, "v": 1}], key_config("id")).passed  # SQLite stores booleans as 0/1
    assert not reconcile([{"id": 1, "v": True}], [{"id": 1, "v": 3}], key_config("id", default_numeric_tolerance=5)).passed


def test_datetime_handling_is_explicit() -> None:
    naive = dt.datetime(2024, 1, 1, 12, 0, 0)
    aware = naive.replace(tzinfo=dt.UTC)
    one, two = [{"id": 1, "t": naive}], [{"id": 1, "t": aware}]

    assert not reconcile(one, two, key_config("id")).passed  # naive vs aware is never silently equal
    assert reconcile(one, two, key_config("id", naive_datetimes_as_utc=True)).passed
    assert reconcile(one, one, key_config("id")).passed  # both naive
    late = [{"id": 1, "t": naive + dt.timedelta(seconds=2)}]
    assert not reconcile(one, late, key_config("id")).passed
    assert reconcile(one, late, key_config("id", tolerances={"t": 2})).passed


def test_normalizers_apply_to_both_sides() -> None:
    config = key_config("id", normalizers={"email": str.lower})

    assert reconcile([{"id": 1, "email": "A@B.test"}], [{"id": 1, "email": "a@b.TEST"}], config).passed
    assert reconcile([{"id": 1, "email": None}], [{"id": 1, "email": None}], config).passed


def test_duplicate_keys_are_counted_never_ignored() -> None:
    result = reconcile([{"id": 1}, {"id": 1}], [{"id": 1}, {"id": 1}, {"id": 1}], key_config("id"))

    assert (result.duplicate_expected_keys, result.duplicate_actual_keys) == (1, 2)
    assert not result.passed and "duplicate keys" in result.summary()


def test_samples_are_bounded_but_counts_are_complete() -> None:
    expected = [{"id": n, "v": 0} for n in range(100)]
    actual = [{"id": n, "v": 1} for n in range(100)]

    result = reconcile(expected, actual, key_config("id", max_samples=3))

    assert result.mismatched_count == 100 and len(result.mismatches) == 3


def test_summaries_mask_sensitive_columns_and_truncate_values() -> None:
    expected = [{"id": 1, "password": "hunter2", "note": "n" * 500}]
    actual = [{"id": 1, "password": "hunter3", "note": "m" * 500}]

    text = reconcile(expected, actual, key_config("id")).summary(Redactor())

    assert "hunter2" not in text and "hunter3" not in text and "***" in text
    assert "n" * 100 not in text and "..." in text


def test_ordered_reconciliation_matches_the_in_memory_result() -> None:
    expected = [{"id": n, "v": n} for n in (1, 2, 3, 5, 8)]
    actual = [{"id": 1, "v": 1}, {"id": 3, "v": 99}, {"id": 4, "v": 4}, {"id": 8, "v": 8}]
    config = key_config("id")

    by_dict, merged = reconcile(expected, actual, config), reconcile_ordered(iter(expected), iter(actual), config)

    for field in ("matched", "missing_count", "unexpected_count", "mismatched_count", "expected_count", "actual_count"):
        assert getattr(merged, field) == getattr(by_dict, field), field
    assert sorted(merged.missing) == sorted(by_dict.missing) and merged.unexpected == [(4,)]


def test_ordered_reconciliation_streams_without_materializing() -> None:
    consumed = 0

    def stream(n: int):  # a generator: only one record is alive at a time
        nonlocal consumed
        for i in range(n):
            consumed += 1
            yield {"id": i, "v": i}

    result = reconcile_ordered(stream(5000), stream(5000), key_config("id"))

    assert result.passed and result.matched == 5000 and consumed == 10000


def test_ordered_reconciliation_refuses_unordered_input() -> None:
    ordered, shuffled = [{"id": 1}, {"id": 2}, {"id": 3}], [{"id": 2}, {"id": 1}, {"id": 3}]

    with pytest.raises(ReconciliationError, match="not ordered by the key"):
        reconcile_ordered(ordered, shuffled, key_config("id"))
    with pytest.raises(ReconciliationError, match="cannot be compared"):
        reconcile_ordered([{"id": 1}], [{"id": "a"}], key_config("id"))
