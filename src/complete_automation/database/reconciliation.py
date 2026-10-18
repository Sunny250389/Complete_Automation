"""Generic record reconciliation between an expected and an actual dataset.

Nothing is matched implicitly: the comparison **key** is explicit, and ignored
fields, numeric tolerances and date/time handling are explicit configuration.
Missing, unexpected, mismatched and duplicate-key records are reported
separately, with counts and a bounded sample; nothing is silently ignored.

Two modes:

- :func:`reconcile` keeps the expected side in a dict (memory proportional to
  that side). Fine for test-sized data.
- :func:`reconcile_ordered` merges two streams that are both ordered by the key
  in the same order, in constant memory. It verifies the ordering and raises
  :class:`ReconciliationError` instead of returning a wrong answer. Use it with
  :func:`reconcile_queries` for large tables. Mind collation: an ``ORDER BY`` on
  text columns can sort differently in two engines, which the ordering check
  then reports.

The module knows nothing about APIs or business rules; a future integration
phase can feed it API-derived rows and database rows.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor
from complete_automation.database.errors import DataAssertionError, ReconciliationError

if TYPE_CHECKING:
    from complete_automation.database.executor import Params, Statement
    from complete_automation.database.session import DatabaseSession

Record = Mapping[str, Any]
Key = tuple[Any, ...]
_VALUE_LIMIT = 60


@dataclass(frozen=True)
class ReconciliationConfig:
    key: tuple[str, ...]
    """Columns that identify a record. Required; there is no default."""
    ignore_fields: frozenset[str] = frozenset()
    compare_fields: tuple[str, ...] | None = None
    """Limit the comparison to these fields; ``None`` compares every field present on either side."""
    tolerances: Mapping[str, float | Decimal] = field(default_factory=dict)
    """Absolute tolerance per field; numbers differ if further apart, datetimes by that many seconds."""
    default_numeric_tolerance: float | Decimal = 0
    naive_datetimes_as_utc: bool = False
    """Treat naive datetimes as UTC before comparing with aware ones. Off: naive vs aware is a mismatch."""
    normalizers: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)
    """Per-field functions applied to both sides first (for example ``str.strip`` or ``str.lower``)."""
    max_samples: int = 10

    def __post_init__(self) -> None:
        if not self.key:
            raise ReconciliationError("reconciliation needs an explicit key")
        if len(set(self.key)) != len(self.key):
            raise ReconciliationError("reconciliation key has duplicate column names")


@dataclass(frozen=True)
class FieldMismatch:
    field: str
    expected: Any
    actual: Any


@dataclass(frozen=True)
class RecordMismatch:
    key: Key
    fields: tuple[FieldMismatch, ...]


@dataclass
class ReconciliationResult:
    config: ReconciliationConfig
    expected_count: int = 0
    actual_count: int = 0
    matched: int = 0
    missing_count: int = 0
    """Expected records absent from the actual dataset."""
    unexpected_count: int = 0
    """Actual records absent from the expected dataset."""
    mismatched_count: int = 0
    duplicate_expected_keys: int = 0
    duplicate_actual_keys: int = 0
    missing: list[Key] = field(default_factory=list)
    unexpected: list[Key] = field(default_factory=list)
    mismatches: list[RecordMismatch] = field(default_factory=list)
    """A bounded sample (``config.max_samples``); ``mismatched_count`` is the full count."""

    @property
    def passed(self) -> bool:
        return not (
            self.missing_count
            or self.unexpected_count
            or self.mismatched_count
            or self.duplicate_expected_keys
            or self.duplicate_actual_keys
        )

    def summary(self, redactor: Redactor = DEFAULT_REDACTOR) -> str:
        parts = [
            f"expected {self.expected_count}, actual {self.actual_count}, matched {self.matched}",
            f"missing {self.missing_count}",
            f"unexpected {self.unexpected_count}",
            f"mismatched {self.mismatched_count}",
        ]
        if self.duplicate_expected_keys or self.duplicate_actual_keys:
            parts.append(
                f"duplicate keys (expected side {self.duplicate_expected_keys}, actual side {self.duplicate_actual_keys})"
            )
        lines = ["reconciliation " + ("passed: " if self.passed else "FAILED: ") + ", ".join(parts)]
        if self.missing:
            lines.append("  missing keys: " + ", ".join(_key_text(k, self.config, redactor) for k in self.missing))
        if self.unexpected:
            lines.append("  unexpected keys: " + ", ".join(_key_text(k, self.config, redactor) for k in self.unexpected))
        for m in self.mismatches:
            fields = "; ".join(
                f"{f.field}: expected {_value_text(f.field, f.expected, redactor)}, actual {_value_text(f.field, f.actual, redactor)}"
                for f in m.fields
            )
            lines.append(f"  {_key_text(m.key, self.config, redactor)} -> {fields}")
        return "\n".join(lines)

    def assert_matches(self, redactor: Redactor = DEFAULT_REDACTOR) -> None:
        if not self.passed:
            raise DataAssertionError(self.summary(redactor))


def _key_text(key: Key, config: ReconciliationConfig, redactor: Redactor) -> str:
    return "(" + ", ".join(f"{n}={_value_text(n, v, redactor)}" for n, v in zip(config.key, key, strict=True)) + ")"


def _value_text(name: str, value: Any, redactor: Redactor) -> str:
    if value is not None and redactor.is_sensitive_key(name):
        return redactor.mask
    text = repr(value)
    return text if len(text) <= _VALUE_LIMIT else text[: _VALUE_LIMIT - 3] + "..."


# --- comparison ---------------------------------------------------------------------------------


def _key_of(record: Record, config: ReconciliationConfig) -> Key:
    try:
        return tuple(record[name] for name in config.key)
    except KeyError as error:
        raise ReconciliationError(f"record is missing key column {error.args[0]!r}") from None


def _field_names(expected: Record, actual: Record, config: ReconciliationConfig) -> list[str]:
    if config.compare_fields is not None:
        names = list(config.compare_fields)
    else:
        names = list(dict.fromkeys([*expected, *actual]))
    skip = set(config.key) | set(config.ignore_fields)
    return [n for n in names if n not in skip]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _aware(value: dt.datetime, config: ReconciliationConfig) -> dt.datetime | None:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=dt.UTC) if config.naive_datetimes_as_utc else None


def _same(name: str, expected: Any, actual: Any, config: ReconciliationConfig) -> bool:
    normalize = config.normalizers.get(name)
    if normalize is not None:
        expected = normalize(expected) if expected is not None else None
        actual = normalize(actual) if actual is not None else None
    tolerance = config.tolerances.get(name)
    if _number(expected) and _number(actual):
        limit = Decimal(str(tolerance if tolerance is not None else config.default_numeric_tolerance))
        return abs(Decimal(str(expected)) - Decimal(str(actual))) <= limit
    if isinstance(expected, dt.datetime) and isinstance(actual, dt.datetime):
        left, right = _aware(expected, config), _aware(actual, config)
        if left is None or right is None:
            if expected.tzinfo is None and actual.tzinfo is None:
                left, right = expected, actual  # both naive: compare as they are
            else:
                return False
        return abs((left - right).total_seconds()) <= float(tolerance or 0)
    return bool(expected == actual)


def compare_records(expected: Record, actual: Record, config: ReconciliationConfig) -> tuple[FieldMismatch, ...]:
    """Field-level differences between two records with the same key (empty means equal)."""
    found = []
    for name in _field_names(expected, actual, config):
        if name not in expected:
            found.append(FieldMismatch(name, "<absent>", actual.get(name)))
        elif name not in actual:
            found.append(FieldMismatch(name, expected[name], "<absent>"))
        elif not _same(name, expected[name], actual[name], config):
            found.append(FieldMismatch(name, expected[name], actual[name]))
    return tuple(found)


class _Builder:
    def __init__(self, config: ReconciliationConfig) -> None:
        self.result = ReconciliationResult(config)
        self.config = config

    def missing(self, key: Key) -> None:
        self.result.missing_count += 1
        if len(self.result.missing) < self.config.max_samples:
            self.result.missing.append(key)

    def unexpected(self, key: Key) -> None:
        self.result.unexpected_count += 1
        if len(self.result.unexpected) < self.config.max_samples:
            self.result.unexpected.append(key)

    def pair(self, key: Key, expected: Record, actual: Record) -> None:
        diffs = compare_records(expected, actual, self.config)
        if diffs:
            self.result.mismatched_count += 1
            if len(self.result.mismatches) < self.config.max_samples:
                self.result.mismatches.append(RecordMismatch(key, diffs))
        else:
            self.result.matched += 1


def reconcile(expected: Iterable[Record], actual: Iterable[Record], config: ReconciliationConfig) -> ReconciliationResult:
    """Compare two datasets by key. Duplicate keys are counted and only the first record per key is compared."""
    builder = _Builder(config)
    result = builder.result
    expected_by_key: dict[Key, Record] = {}
    for record in expected:
        result.expected_count += 1
        key = _key_of(record, config)
        if key in expected_by_key:
            result.duplicate_expected_keys += 1
        else:
            expected_by_key[key] = record
    seen: set[Key] = set()
    for record in actual:
        result.actual_count += 1
        key = _key_of(record, config)
        if key in seen:
            result.duplicate_actual_keys += 1
            continue
        seen.add(key)
        if key not in expected_by_key:
            builder.unexpected(key)
        else:
            builder.pair(key, expected_by_key[key], record)
    for key in expected_by_key:
        if key not in seen:
            builder.missing(key)
    return result


def reconcile_ordered(
    expected: Iterable[Record], actual: Iterable[Record], config: ReconciliationConfig
) -> ReconciliationResult:
    """Constant-memory comparison of two streams that are both ordered by ``config.key``, ascending.

    Raises:
        ReconciliationError: a stream is not strictly ordered by the key (or keys are not comparable).
    """
    builder = _Builder(config)
    result = builder.result
    left, right = _ordered(iter(expected), config, "expected", result), _ordered(iter(actual), config, "actual", result)
    a, b = next(left, None), next(right, None)
    while a is not None or b is not None:
        if b is None or (a is not None and _less(a[0], b[0])):
            assert a is not None
            builder.missing(a[0])
            a = next(left, None)
        elif a is None or _less(b[0], a[0]):
            builder.unexpected(b[0])
            b = next(right, None)
        else:
            builder.pair(a[0], a[1], b[1])
            a, b = next(left, None), next(right, None)
    return result


def _less(left: Key, right: Key) -> bool:
    try:
        return left < right
    except TypeError:
        raise ReconciliationError("key values of the two datasets cannot be compared (different types?)") from None


def _ordered(
    records: Iterator[Record], config: ReconciliationConfig, side: str, result: ReconciliationResult
) -> Iterator[tuple[Key, Record]]:
    previous: Key | None = None
    for record in records:
        key = _key_of(record, config)
        if side == "expected":
            result.expected_count += 1
        else:
            result.actual_count += 1
        if previous is not None:
            if key == previous:
                if side == "expected":
                    result.duplicate_expected_keys += 1
                else:
                    result.duplicate_actual_keys += 1
                continue
            if _less(key, previous):
                raise ReconciliationError(
                    f"the {side} records are not ordered by the key {config.key}: "
                    "order both sides by the key (same direction and collation)"
                )
        previous = key
        yield key, record


def reconcile_queries(
    expected_db: DatabaseSession,
    expected_statement: Statement,
    actual_db: DatabaseSession,
    actual_statement: Statement,
    config: ReconciliationConfig,
    *,
    expected_params: Params = None,
    actual_params: Params = None,
    chunk_size: int = 1000,
) -> ReconciliationResult:
    """Reconcile two queries (possibly on different databases) in constant memory.

    Both statements must ``ORDER BY`` the key columns ascending. The two
    sessions must be different connections when both streams are open at once.
    """
    if expected_db is actual_db:
        raise ReconciliationError("use two sessions: both result streams are open at the same time")
    return reconcile_ordered(
        expected_db.iterate(expected_statement, expected_params, chunk_size=chunk_size),
        actual_db.iterate(actual_statement, actual_params, chunk_size=chunk_size),
        config,
    )


def key_config(*key: str, **options: Any) -> ReconciliationConfig:
    """``key_config("id", ignore_fields=frozenset({"updated_at"}))`` as shorthand."""
    return ReconciliationConfig(key=tuple(key), **options)


def table_rows(db: DatabaseSession, statement: Statement, params: Params = None) -> Sequence[Record]:
    """Convenience: a query's rows as a list, for :func:`reconcile`."""
    return db.query(statement, params).rows
