"""Lightweight migration checks: compare schema snapshots and row counts, read a schema version.

This module *validates*; it never runs a migration. Apply migrations with your
migration tool (for example Alembic) against a database you control, take a
snapshot before and after, and use these helpers to assert that exactly the
declared changes happened and that no data went missing. Nothing here alters a
database, and a failed check does not repair anything.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from complete_automation.database.errors import DataAssertionError, SchemaValidationError
from complete_automation.database.validations.assertions import count_rows
from complete_automation.database.validations.identifiers import table_ref, typed_column
from complete_automation.database.validations.schema import ColumnExpectation, SchemaExpectation

if TYPE_CHECKING:
    from complete_automation.database.session import DatabaseSession


class SchemaChanges(BaseModel):
    """What differs between two snapshots, keyed by table."""

    model_config = ConfigDict(frozen=True)

    added_tables: tuple[str, ...] = ()
    removed_tables: tuple[str, ...] = ()
    added_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    removed_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    changed_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Columns whose type, nullability, length or default changed."""

    @property
    def empty(self) -> bool:
        return not (
            self.added_tables or self.removed_tables or self.added_columns or self.removed_columns or self.changed_columns
        )


def _signature(column: ColumnExpectation) -> tuple[object, ...]:
    return (column.type, column.nullable, column.max_length, column.default, column.primary_key)


def diff_snapshots(before: SchemaExpectation, after: SchemaExpectation) -> SchemaChanges:
    """Compare two snapshots from :func:`snapshot_schema`."""
    old = {t.name: {c.name: c for c in t.columns} for t in before.tables}
    new = {t.name: {c.name: c for c in t.columns} for t in after.tables}
    added_columns: dict[str, tuple[str, ...]] = {}
    removed_columns: dict[str, tuple[str, ...]] = {}
    changed_columns: dict[str, tuple[str, ...]] = {}
    for table in sorted(old.keys() & new.keys()):
        if added := tuple(sorted(new[table].keys() - old[table].keys())):
            added_columns[table] = added
        if removed := tuple(sorted(old[table].keys() - new[table].keys())):
            removed_columns[table] = removed
        if changed := tuple(
            sorted(n for n in old[table].keys() & new[table].keys() if _signature(old[table][n]) != _signature(new[table][n]))
        ):
            changed_columns[table] = changed
    return SchemaChanges(
        added_tables=tuple(sorted(new.keys() - old.keys())),
        removed_tables=tuple(sorted(old.keys() - new.keys())),
        added_columns=added_columns,
        removed_columns=removed_columns,
        changed_columns=changed_columns,
    )


def assert_schema_changes(
    before: SchemaExpectation,
    after: SchemaExpectation,
    *,
    added_tables: Collection[str] = (),
    removed_tables: Collection[str] = (),
    added_columns: Mapping[str, Collection[str]] | None = None,
    removed_columns: Mapping[str, Collection[str]] | None = None,
    changed_columns: Mapping[str, Collection[str]] | None = None,
) -> SchemaChanges:
    """Assert the migration changed exactly what is declared: nothing missing, nothing extra."""
    actual = diff_snapshots(before, after)
    wanted = SchemaChanges(
        added_tables=tuple(sorted(added_tables)),
        removed_tables=tuple(sorted(removed_tables)),
        added_columns={t: tuple(sorted(c)) for t, c in (added_columns or {}).items()},
        removed_columns={t: tuple(sorted(c)) for t, c in (removed_columns or {}).items()},
        changed_columns={t: tuple(sorted(c)) for t, c in (changed_columns or {}).items()},
    )
    problems = [
        f"{label}: expected {getattr(wanted, label)}, actual {getattr(actual, label)}"
        for label in ("added_tables", "removed_tables", "added_columns", "removed_columns", "changed_columns")
        if getattr(wanted, label) != getattr(actual, label)
    ]
    if problems:
        raise SchemaValidationError("migration changed the schema differently than declared:\n  - " + "\n  - ".join(problems))
    return actual


def table_counts(db: DatabaseSession, tables: Sequence[str], *, schema: str | None = None) -> dict[str, int]:
    """Row counts per table, for before/after comparison around a migration."""
    return {name: count_rows(db, name, schema=schema) for name in tables}


def assert_counts_preserved(
    before: Mapping[str, int], after: Mapping[str, int], *, allowed_changes: Mapping[str, int] | None = None
) -> None:
    """Assert each table has the same count after as before, except by the declared delta."""
    allowed = allowed_changes or {}
    problems = []
    for name, count in before.items():
        expected = count + allowed.get(name, 0)
        actual = after.get(name)
        if actual != expected:
            problems.append(f"{name}: expected {expected}, found {actual}")
    if problems:
        raise DataAssertionError("row counts changed unexpectedly: " + "; ".join(problems))


def read_schema_version(
    db: DatabaseSession, *, table: str = "alembic_version", column: str = "version_num", schema: str | None = None
) -> str | None:
    """The version recorded by a migration tool (Alembic's defaults), or ``None`` if empty."""
    rows = db.query(select(typed_column(column)).select_from(table_ref(table, schema)).limit(2)).rows
    if len(rows) > 1:
        raise DataAssertionError(f"{table} holds more than one version row")
    return str(rows[0][column]) if rows else None


def assert_schema_version(
    db: DatabaseSession,
    expected: str,
    *,
    table: str = "alembic_version",
    column: str = "version_num",
    schema: str | None = None,
) -> None:
    actual = read_schema_version(db, table=table, column=column, schema=schema)
    if actual != expected:
        raise DataAssertionError(f"schema version: expected {expected!r}, found {actual!r}")
