"""Referential integrity and cross-table consistency checks.

Generic by design: callers name the tables and columns; nothing here knows a
business schema. Engines differ in what they enforce (SQLite ignores foreign
keys unless the pragma is on), so :func:`assert_foreign_keys_enforced` exists
and the cascade/restrict checks run against the real engine instead of
assuming behavior.

The cascade and restrict checks write: they delete inside a savepoint that is
always rolled back, so they leave the data as they found it, but they need
write access and a savepoint-capable engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import ColumnElement, delete, func, select

from complete_automation.database.errors import DataAssertionError, DatabaseIntegrityError
from complete_automation.database.validations.assertions import (
    aggregate,
    describe_where,
    show,
    values_equal,
)
from complete_automation.database.validations.identifiers import (
    conditions,
    table_ref,
    table_with_columns,
    typed_column,
)

if TYPE_CHECKING:
    from complete_automation.database.session import DatabaseSession

_SAMPLE_LIMIT = 5


# --- orphans ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrphanReport:
    child: str
    parent: str
    count: int
    sample: tuple[Any, ...]
    """A few orphaned foreign-key values (only reported when asked for)."""


def find_orphans(
    db: DatabaseSession,
    *,
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
    child_schema: str | None = None,
    parent_schema: str | None = None,
    limit: int = _SAMPLE_LIMIT,
) -> OrphanReport:
    """Child rows whose non-NULL ``child_column`` has no matching ``parent_column`` row."""
    child = table_with_columns(child_table, [child_column], child_schema).alias("c")
    parent = table_with_columns(parent_table, [parent_column], parent_schema).alias("p")
    base = (
        select(child.c[child_column])
        .select_from(child.outerjoin(parent, child.c[child_column] == parent.c[parent_column]))
        .where(child.c[child_column].is_not(None), parent.c[parent_column].is_(None))
    )
    total = int(db.scalar(select(func.count()).select_from(base.subquery())))
    sample = tuple(db.query(base.limit(limit)).column(child_column)) if total else ()
    return OrphanReport(child=f"{child_table}.{child_column}", parent=f"{parent_table}.{parent_column}", count=total, sample=sample)


def assert_no_orphans(
    db: DatabaseSession,
    *,
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
    child_schema: str | None = None,
    parent_schema: str | None = None,
    show_values: bool = False,
) -> None:
    report = find_orphans(
        db,
        child_table=child_table,
        child_column=child_column,
        parent_table=parent_table,
        parent_column=parent_column,
        child_schema=child_schema,
        parent_schema=parent_schema,
    )
    if report.count:
        detail = ""
        if show_values:
            detail = ": " + ", ".join(show(db, child_column, v) for v in report.sample)
        raise DataAssertionError(
            f"orphaned rows: {report.count} row(s) in {report.child} have no matching {report.parent}{detail}"
        )


def assert_foreign_keys_enforced(db: DatabaseSession) -> None:
    """Assert the connection enforces foreign keys (SQLite needs ``PRAGMA foreign_keys = ON``)."""
    if not db.manager.adapter.foreign_keys_enforced(db.connection):
        raise DataAssertionError(
            f"foreign keys are not enforced on this {db.manager.dialect} connection; "
            "constraint tests would pass for the wrong reason"
        )


# --- cascade and restrict ---------------------------------------------------------------------


def assert_delete_restricted(
    db: DatabaseSession, table: str, where: Mapping[str, Any], *, schema: str | None = None
) -> None:
    """Assert deleting the matching rows is rejected by a foreign key. The attempt is rolled back."""
    if not where:
        raise DataAssertionError("assert_delete_restricted needs a condition (it would delete every row)")
    with db.rolled_back_savepoint():
        try:
            db.execute(delete(table_ref(table, schema)).where(*conditions(where)))
        except DatabaseIntegrityError:
            return
        raise DataAssertionError(
            f"deleting from {table}{describe_where(db, where)} was expected to be restricted by a foreign key, but it succeeded"
        )


@dataclass(frozen=True)
class Dependent:
    """A child table that references the parent being deleted."""

    table: str
    column: str
    """Foreign-key column in the child table."""
    parent_column: str
    """Referenced column in the parent table."""
    schema: str | None = None


def assert_delete_cascades(
    db: DatabaseSession,
    parent_table: str,
    where: Mapping[str, Any],
    dependents: Sequence[Dependent],
    *,
    parent_schema: str | None = None,
) -> None:
    """Assert deleting parents removes exactly their dependents and nothing else. Rolled back afterwards."""
    if not where:
        raise DataAssertionError("assert_delete_cascades needs a condition (it would delete every row)")
    parent = table_ref(parent_table, parent_schema)
    with db.rolled_back_savepoint():
        before = {d: _dependent_counts(db, parent, where, d) for d in dependents}
        deleted = db.execute(delete(parent).where(*conditions(where))).rowcount
        if not deleted:
            raise DataAssertionError(f"no {parent_table} row matched{describe_where(db, where)}")
        problems = []
        for dep in dependents:
            owned_before, total_before = before[dep]
            owned_after, total_after = _dependent_counts(db, parent, where, dep)
            label = f"{dep.table}.{dep.column}"
            if total_after != total_before - owned_before:
                problems.append(
                    f"{label}: expected {total_before - owned_before} row(s) left, found {total_after}"
                )
            if owned_after:
                problems.append(f"{label}: {owned_after} dependent row(s) remain")
    if problems:
        raise DataAssertionError(f"cascade from {parent_table}{describe_where(db, where)} is wrong: " + "; ".join(problems))


def _dependent_counts(db: DatabaseSession, parent: Any, where: Mapping[str, Any], dep: Dependent) -> tuple[int, int]:
    """(rows that belong to the parents matching ``where``, all rows) of the dependent table."""
    child = table_ref(dep.table, dep.schema)
    keys = select(typed_column(dep.parent_column)).select_from(parent).where(*conditions(where))
    owned = select(func.count()).select_from(child).where(typed_column(dep.column).in_(keys))
    total = select(func.count()).select_from(child)
    return int(db.scalar(owned)), int(db.scalar(total))


# --- derived fields and source/target reconciliation ------------------------------------------------------------


@dataclass(frozen=True)
class DerivedMismatch:
    key: Any
    stored: Any
    computed: Any


def find_derived_mismatches(
    db: DatabaseSession,
    *,
    parent_table: str,
    parent_key: str,
    parent_column: str,
    child_table: str,
    child_key: str,
    child_column: str | None = None,
    function: Literal["sum", "count", "min", "max"] = "sum",
    tolerance: float | Decimal = 0,
    parent_schema: str | None = None,
    child_schema: str | None = None,
    limit: int = 1000,
) -> list[DerivedMismatch]:
    """Parents whose stored ``parent_column`` differs from ``function`` over their children.

    Example: ``orders.total`` against ``sum(order_items.amount)`` grouped by ``order_id``.
    Parents without children are computed as 0 for ``sum``/``count``, and as NULL for ``min``/``max``.
    """
    needed_child = [child_key] + ([child_column] if child_column else [])
    parent = table_with_columns(parent_table, [parent_key, parent_column], parent_schema)
    child = table_with_columns(child_table, needed_child, child_schema)
    computed: ColumnElement[Any]
    if function == "count":
        computed = func.count(child.c[child_key])
    elif child_column is None:
        raise DataAssertionError(f"{function} needs child_column")
    else:
        computed = getattr(func, function)(child.c[child_column])
        if function == "sum":
            computed = func.coalesce(computed, 0)
    statement = (
        select(parent.c[parent_key].label("key"), parent.c[parent_column].label("stored"), computed.label("computed"))
        .select_from(parent.outerjoin(child, child.c[child_key] == parent.c[parent_key]))
        .group_by(parent.c[parent_key], parent.c[parent_column])
        .order_by(parent.c[parent_key])
    )
    found: list[DerivedMismatch] = []
    for row in db.iterate(statement):
        if not values_equal(row["stored"], row["computed"], tolerance):
            found.append(DerivedMismatch(row["key"], row["stored"], row["computed"]))
            if len(found) >= limit:
                break
    return found


def assert_derived_consistent(db: DatabaseSession, **kwargs: Any) -> None:
    """Like :func:`find_derived_mismatches`, failing when any parent's stored value is off."""
    mismatches = find_derived_mismatches(db, **kwargs)
    if mismatches:
        parent = f"{kwargs['parent_table']}.{kwargs['parent_column']}"
        key_name = kwargs["parent_key"]
        shown = "; ".join(
            f"{key_name}={show(db, key_name, m.key)}: stored {show(db, kwargs['parent_column'], m.stored)}, "
            f"computed {show(db, kwargs['parent_column'], m.computed)}"
            for m in mismatches[:_SAMPLE_LIMIT]
        )
        raise DataAssertionError(f"derived field {parent} is inconsistent in {len(mismatches)} row(s): {shown}")


def assert_aggregate_matches(
    db: DatabaseSession,
    *,
    source_table: str,
    target_table: str,
    function: Literal["count", "sum", "avg", "min", "max"] = "count",
    source_column: str | None = None,
    target_column: str | None = None,
    source_where: Mapping[str, Any] | None = None,
    target_where: Mapping[str, Any] | None = None,
    source_schema: str | None = None,
    target_schema: str | None = None,
    tolerance: float | Decimal = 0,
) -> None:
    """Assert an aggregate over a source table equals the same aggregate over a target table."""
    source = aggregate(db, source_table, function, source_column, source_where, schema=source_schema)
    target = aggregate(db, target_table, function, target_column or source_column, target_where, schema=target_schema)
    if not values_equal(source, target, tolerance):
        what = f"{function}({source_column or '*'})"
        raise DataAssertionError(
            f"{what} differs: {source_table}{describe_where(db, source_where)} = {source!r}, "
            f"{target_table}{describe_where(db, target_where)} = {target!r}"
        )
