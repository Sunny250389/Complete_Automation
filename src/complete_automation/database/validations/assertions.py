"""Data assertions with actionable, safe diagnostics.

Every helper takes a :class:`DatabaseSession`, a table name (and optional
schema), builds its SQL with SQLAlchemy expressions and bound values, and
raises :class:`DataAssertionError` (an ``AssertionError``) on failure. The
message names the check, the table and columns, what was expected and what was
found. It never dumps whole rows: only the compared columns appear, sensitive
column names are masked by the shared redactor, and long values are truncated.
Pass ``show_values=True`` where offered to include key values.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import Select, func, literal_column, or_, select
from sqlalchemy.sql.elements import ColumnElement

from complete_automation.database.errors import DataAssertionError
from complete_automation.database.validations.identifiers import (
    column_names,
    conditions,
    safe_identifier,
    table_ref,
    typed_column,
)

if TYPE_CHECKING:
    from complete_automation.database.session import DatabaseSession

Where = Mapping[str, Any] | None
_VALUE_LIMIT = 80
_SAMPLE_LIMIT = 5


def show(db: DatabaseSession, column: str, value: Any) -> str:
    """A short, redacted rendering of a column value for failure messages."""
    if value is not None and db.manager.redactor.is_sensitive_key(column):
        return db.manager.redactor.mask
    text = repr(value)
    return text if len(text) <= _VALUE_LIMIT else text[: _VALUE_LIMIT - 3] + "..."


def describe_where(db: DatabaseSession, where: Where) -> str:
    if not where:
        return ""
    parts = ", ".join(f"{name}={show(db, name, value)}" for name, value in where.items())
    return f" where {parts}"


def _label(table: str, schema: str | None) -> str:
    return f"{schema}.{table}" if schema else table


def _fail(message: str) -> DataAssertionError:
    return DataAssertionError(message)


def values_equal(expected: Any, actual: Any, tolerance: float | Decimal = 0) -> bool:
    """Equality that allows an absolute numeric ``tolerance`` (never for bool or non-numbers)."""
    if tolerance and _is_number(expected) and _is_number(actual):
        return abs(Decimal(str(expected)) - Decimal(str(actual))) <= Decimal(str(tolerance))
    if _is_number(expected) and _is_number(actual):
        return Decimal(str(expected)) == Decimal(str(actual))
    return bool(expected == actual)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


# --- counts ---------------------------------------------------------------------


def count_rows(db: DatabaseSession, table: str, where: Where = None, *, schema: str | None = None) -> int:
    statement = select(func.count()).select_from(table_ref(table, schema)).where(*conditions(where))
    return int(db.scalar(statement))


def assert_row_count(
    db: DatabaseSession, table: str, expected: int, where: Where = None, *, schema: str | None = None
) -> None:
    actual = count_rows(db, table, where, schema=schema)
    if actual != expected:
        raise _fail(
            f"row count of {_label(table, schema)}{describe_where(db, where)}: expected {expected}, found {actual}"
        )


def assert_row_count_between(
    db: DatabaseSession,
    table: str,
    minimum: int,
    maximum: int,
    where: Where = None,
    *,
    schema: str | None = None,
) -> None:
    actual = count_rows(db, table, where, schema=schema)
    if not minimum <= actual <= maximum:
        raise _fail(
            f"row count of {_label(table, schema)}{describe_where(db, where)}: "
            f"expected {minimum}..{maximum}, found {actual}"
        )


def assert_table_empty(db: DatabaseSession, table: str, *, schema: str | None = None) -> None:
    assert_row_count(db, table, 0, schema=schema)


def assert_table_not_empty(db: DatabaseSession, table: str, *, schema: str | None = None) -> None:
    if count_rows(db, table, schema=schema) == 0:
        raise _fail(f"{_label(table, schema)}: expected at least one row, found none")


# --- records --------------------------------------------------------------------


def assert_row_exists(
    db: DatabaseSession, table: str, where: Mapping[str, Any], *, schema: str | None = None
) -> dict[str, Any]:
    """Assert exactly one row matches ``where`` and return it (all columns)."""
    if not where:
        raise _fail("assert_row_exists needs at least one condition")
    return _fetch_single(db, table, where, schema)


def _fetch_single(db: DatabaseSession, table: str, where: Mapping[str, Any], schema: str | None) -> dict[str, Any]:
    statement: Select[Any] = (
        select(literal_column("*")).select_from(table_ref(table, schema)).where(*conditions(where)).limit(2)
    )
    rows = db.query(statement).rows
    if not rows:
        raise _fail(f"{_label(table, schema)}: no row found{describe_where(db, where)}")
    if len(rows) > 1:
        raise _fail(f"{_label(table, schema)}: expected one row{describe_where(db, where)}, found more than one")
    return rows[0]


def assert_row_absent(db: DatabaseSession, table: str, where: Mapping[str, Any], *, schema: str | None = None) -> None:
    if not where:
        raise _fail("assert_row_absent needs at least one condition")
    found = count_rows(db, table, where, schema=schema)
    if found:
        raise _fail(f"{_label(table, schema)}: expected no row{describe_where(db, where)}, found {found}")


def assert_record(
    db: DatabaseSession,
    table: str,
    key: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tolerance: float | Decimal = 0,
    schema: str | None = None,
) -> dict[str, Any]:
    """Assert the row identified by ``key`` has the ``expected`` column values; return the row."""
    row = _fetch_single(db, table, key, schema)
    problems = []
    for name, want in expected.items():
        safe_identifier(name, what="column name")
        if name not in row:
            problems.append(f"{name}: column does not exist")
        elif not values_equal(want, row[name], tolerance):
            problems.append(f"{name}: expected {show(db, name, want)}, found {show(db, name, row[name])}")
    if problems:
        raise _fail(
            f"{_label(table, schema)}{describe_where(db, key)}: {len(problems)} field(s) differ: "
            + "; ".join(problems)
        )
    return row


def assert_required_fields(record: Mapping[str, Any], fields: Iterable[str], *, label: str = "record") -> None:
    """Assert each field is present and not ``None`` (empty strings count as missing too)."""
    missing = [f for f in fields if f not in record or record[f] is None or record[f] == ""]
    if missing:
        raise _fail(f"{label}: required field(s) missing or empty: {', '.join(missing)}")


def assert_not_null(
    db: DatabaseSession,
    table: str,
    columns: str | Sequence[str],
    where: Where = None,
    *,
    schema: str | None = None,
) -> None:
    """Assert no row has NULL in any of ``columns``."""
    names = column_names(columns)
    problems = []
    for name in names:
        statement = (
            select(func.count())
            .select_from(table_ref(table, schema))
            .where(typed_column(name).is_(None), *conditions(where))
        )
        nulls = int(db.scalar(statement))
        if nulls:
            problems.append(f"{name} ({nulls})")
    if problems:
        raise _fail(
            f"{_label(table, schema)}{describe_where(db, where)}: NULL values found in " + ", ".join(problems)
        )


# --- duplicates ----------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateGroup:
    values: tuple[Any, ...]
    count: int


def find_duplicates(
    db: DatabaseSession,
    table: str,
    columns: str | Sequence[str],
    where: Where = None,
    *,
    schema: str | None = None,
    limit: int = 10,
) -> list[DuplicateGroup]:
    """Groups of rows sharing the same values in ``columns`` (NULLs are not treated as equal)."""
    names = column_names(columns)
    cols = [typed_column(n) for n in names]
    statement = (
        select(*cols, func.count().label("duplicate_count"))
        .select_from(table_ref(table, schema))
        .where(*conditions(where), *[c.is_not(None) for c in cols])
        .group_by(*cols)
        .having(func.count() > 1)
        .order_by(func.count().desc())
        .limit(limit)
    )
    return [
        DuplicateGroup(tuple(row[n] for n in names), int(row["duplicate_count"])) for row in db.query(statement).rows
    ]


def assert_no_duplicates(
    db: DatabaseSession,
    table: str,
    columns: str | Sequence[str],
    where: Where = None,
    *,
    schema: str | None = None,
    show_values: bool = False,
) -> None:
    names = column_names(columns)
    groups = find_duplicates(db, table, names, where, schema=schema, limit=_SAMPLE_LIMIT + 1)
    if groups:
        shown = groups[:_SAMPLE_LIMIT]
        more = "+" if len(groups) > _SAMPLE_LIMIT else ""
        detail = ""
        if show_values:
            detail = ": " + "; ".join(
                f"({', '.join(show(db, n, v) for n, v in zip(names, g.values, strict=True))}) x{g.count}"
                for g in shown
            )
        raise _fail(
            f"duplicates in {_label(table, schema)}({', '.join(names)}){describe_where(db, where)}: "
            f"{len(shown)}{more} duplicated value group(s), "
            f"{sum(g.count for g in shown)} rows involved in the first {len(shown)}{detail}"
        )


# --- value rules -------------------------------------------------------------------------


def assert_values_in_range(
    db: DatabaseSession,
    table: str,
    column: str,
    minimum: Any = None,
    maximum: Any = None,
    where: Where = None,
    *,
    schema: str | None = None,
) -> None:
    """Assert every non-NULL ``column`` value lies in ``[minimum, maximum]`` (numbers, dates, timestamps)."""
    if minimum is None and maximum is None:
        raise _fail("assert_values_in_range needs a minimum or a maximum")
    sample = minimum if minimum is not None else maximum
    col = typed_column(column, sample)
    outside: list[ColumnElement[bool]] = []
    if minimum is not None:
        outside.append(col < minimum)
    if maximum is not None:
        outside.append(col > maximum)
    statement = (
        select(func.count()).select_from(table_ref(table, schema)).where(or_(*outside), *conditions(where))
    )
    bad = int(db.scalar(statement))
    if bad:
        bounds = f"[{'-inf' if minimum is None else minimum}, {'+inf' if maximum is None else maximum}]"
        raise _fail(
            f"{_label(table, schema)}.{column}{describe_where(db, where)}: {bad} row(s) outside {bounds}"
        )


def assert_string_length(
    db: DatabaseSession,
    table: str,
    column: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    where: Where = None,
    schema: str | None = None,
) -> None:
    """Assert every non-NULL ``column`` value has a character length within the bounds."""
    if min_length is None and max_length is None:
        raise _fail("assert_string_length needs min_length or max_length")
    col = typed_column(column)
    length = func.length(col)
    outside: list[ColumnElement[bool]] = []
    if min_length is not None:
        outside.append(length < min_length)
    if max_length is not None:
        outside.append(length > max_length)
    statement = (
        select(func.count())
        .select_from(table_ref(table, schema))
        .where(col.is_not(None), or_(*outside), *conditions(where))
    )
    bad = int(db.scalar(statement))
    if bad:
        raise _fail(
            f"{_label(table, schema)}.{column}{describe_where(db, where)}: {bad} row(s) with length outside "
            f"[{min_length if min_length is not None else 0}, {max_length if max_length is not None else 'inf'}]"
        )


def assert_values_in_set(
    db: DatabaseSession,
    table: str,
    column: str,
    allowed: Collection[Any],
    where: Where = None,
    *,
    allow_null: bool = False,
    schema: str | None = None,
) -> None:
    """Assert every value of ``column`` is one of ``allowed`` (NULL only if ``allow_null``)."""
    col = typed_column(column)
    rows = db.query(
        select(col).select_from(table_ref(table, schema)).where(*conditions(where)).distinct()
    ).column(column)
    unexpected = [v for v in rows if v not in allowed and not (v is None and allow_null)]
    if unexpected:
        shown = ", ".join(show(db, column, v) for v in unexpected[:_SAMPLE_LIMIT])
        raise _fail(
            f"{_label(table, schema)}.{column}{describe_where(db, where)}: {len(unexpected)} unexpected "
            f"distinct value(s): {shown}"
        )


def assert_distinct_count(
    db: DatabaseSession,
    table: str,
    column: str,
    expected: int,
    where: Where = None,
    *,
    schema: str | None = None,
) -> None:
    statement = (
        select(func.count(typed_column(column).distinct()))
        .select_from(table_ref(table, schema))
        .where(*conditions(where))
    )
    actual = int(db.scalar(statement))
    if actual != expected:
        raise _fail(
            f"distinct {column} in {_label(table, schema)}{describe_where(db, where)}: expected {expected}, found {actual}"
        )


def assert_grouped_counts(
    db: DatabaseSession,
    table: str,
    group_by: str,
    expected: Mapping[Any, int],
    where: Where = None,
    *,
    allow_extra_groups: bool = False,
    schema: str | None = None,
) -> None:
    """Assert ``count(*)`` per distinct ``group_by`` value equals ``expected`` (missing groups count as 0)."""
    col = typed_column(group_by)
    statement = (
        select(col, func.count().label("group_count"))
        .select_from(table_ref(table, schema))
        .where(*conditions(where))
        .group_by(col)
    )
    actual = {row[group_by]: int(row["group_count"]) for row in db.query(statement).rows}
    problems = [
        f"{show(db, group_by, key)}: expected {count}, found {actual.get(key, 0)}"
        for key, count in expected.items()
        if actual.get(key, 0) != count
    ]
    if not allow_extra_groups:
        problems += [f"{show(db, group_by, key)}: unexpected group ({n} rows)" for key, n in actual.items() if key not in expected]
    if problems:
        raise _fail(
            f"counts of {_label(table, schema)} by {group_by}{describe_where(db, where)}: " + "; ".join(problems)
        )


def aggregate(
    db: DatabaseSession,
    table: str,
    function: Literal["count", "sum", "avg", "min", "max"],
    column: str | None = None,
    where: Where = None,
    *,
    schema: str | None = None,
) -> Any:
    if function not in ("count", "sum", "avg", "min", "max"):
        raise _fail(f"unsupported aggregate function {function!r}")
    if function == "count":
        expression = func.count(typed_column(column)) if column else func.count()
    else:
        if column is None:
            raise _fail(f"{function} needs a column")
        expression = getattr(func, function)(typed_column(column))
    statement = select(expression).select_from(table_ref(table, schema)).where(*conditions(where))
    return db.scalar(statement)


def assert_aggregate(
    db: DatabaseSession,
    table: str,
    function: Literal["count", "sum", "avg", "min", "max"],
    column: str | None,
    expected: Any,
    where: Where = None,
    *,
    tolerance: float | Decimal = 0,
    schema: str | None = None,
) -> None:
    actual = aggregate(db, table, function, column, where, schema=schema)
    if not values_equal(expected, actual, tolerance):
        target = f"{function}({column or '*'})"
        tol = f" (tolerance {tolerance})" if tolerance else ""
        raise _fail(
            f"{target} of {_label(table, schema)}{describe_where(db, where)}: expected {expected!r}{tol}, found {actual!r}"
        )


# --- ordering and pagination ------------------------------------------------------------------


def assert_sorted(rows: Iterable[Mapping[str, Any]], column: str, *, descending: bool = False) -> None:
    """Assert ``rows`` (for example ``QueryResult.rows``) are ordered by ``column``. NULLs are not allowed."""
    values = [row[column] for row in rows]
    if any(v is None for v in values):
        raise _fail(f"ordering by {column}: NULL values cannot be ordered portably; filter them out first")
    for index in range(1, len(values)):
        before, after = values[index - 1], values[index]
        if (before < after) if descending else (before > after):
            raise _fail(
                f"rows are not sorted {'descending' if descending else 'ascending'} by {column}: "
                f"row {index - 1} then row {index} are out of order"
            )


def assert_pagination_consistent(
    db: DatabaseSession,
    table: str,
    order_by: str,
    *,
    page_size: int,
    where: Where = None,
    schema: str | None = None,
) -> int:
    """Page through ``table`` with LIMIT/OFFSET and assert the pages tile the full ordered set.

    ``order_by`` must be unique (for example the primary key), otherwise page
    boundaries are not deterministic. Returns the total row count.
    """
    if page_size < 1:
        raise _fail("page_size must be at least 1")
    col = typed_column(order_by)
    base = select(col).select_from(table_ref(table, schema)).where(*conditions(where)).order_by(col)
    everything = db.query(base).column(order_by)
    if len(set(everything)) != len(everything):
        raise _fail(f"{_label(table, schema)}.{order_by} is not unique: pagination cannot be verified")
    paged: list[Any] = []
    offset = 0
    while True:
        page = db.query(base.limit(page_size).offset(offset)).column(order_by)
        if not page:
            break
        if len(page) > page_size:
            raise _fail(f"page at offset {offset} returned {len(page)} rows, more than page_size {page_size}")
        paged.extend(page)
        offset += page_size
    if paged != everything:
        raise _fail(
            f"pagination of {_label(table, schema)} by {order_by}: pages hold {len(paged)} rows, "
            f"the full ordered set holds {len(everything)}; rows were skipped, repeated or reordered"
        )
    return len(everything)


# --- dates -----------------------------------------------------------------------------------------


def assert_recent(
    db: DatabaseSession,
    table: str,
    column: str,
    *,
    within: dt.timedelta,
    now: dt.datetime,
    where: Where = None,
    schema: str | None = None,
) -> None:
    """Assert ``max(column)`` is no older than ``within`` before the explicit ``now`` (no hidden clock)."""
    latest = aggregate(db, table, "max", column, where, schema=schema)
    if latest is None:
        raise _fail(f"{_label(table, schema)}.{column}{describe_where(db, where)}: no values to check")
    if isinstance(latest, str):
        latest = dt.datetime.fromisoformat(latest)
    if latest.tzinfo is None and now.tzinfo is not None:
        latest = latest.replace(tzinfo=now.tzinfo)
    if now - latest > within:
        raise _fail(
            f"{_label(table, schema)}.{column}: newest value is older than {within} before {now.isoformat()}"
        )
