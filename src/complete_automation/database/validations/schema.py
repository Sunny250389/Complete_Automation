"""Schema inspection and validation against explicit expectations.

An expectation lists what must be true; anything it leaves ``None`` is not
checked, and extra tables or columns are tolerated unless ``exact_tables`` /
``exact_columns`` say otherwise. Two schemas are never "equal because the table
names match": columns, types, nullability, keys, constraints and indexes are
compared one by one, and every difference is reported.

Types are compared as **families** (``integer``, ``numeric``, ``float``,
``string``, ``boolean``, ``date``, ``datetime``, ``time``, ``binary``, ``json``,
``uuid``), because engines name and reflect types differently (SQLite
``VARCHAR(10)`` and PostgreSQL ``character varying(10)`` are both ``string``).
Set ``native_type`` to also require the engine's own reflected type name.

Expectations are pydantic models, so they can be written in YAML and loaded
with the shared ``DataLoader``/``load_yaml``. Validation only reads metadata;
it never alters the database.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import inspect
from sqlalchemy import types as sat
from sqlalchemy.engine import Connection

from complete_automation.database.errors import SchemaValidationError

if TYPE_CHECKING:
    from complete_automation.database.session import DatabaseSession

TypeFamily = Literal[
    "integer", "numeric", "float", "string", "boolean", "date", "datetime", "time", "binary", "json", "uuid", "other"
]  # fmt: skip
_MAX_REPORTED = 50


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnExpectation(_Model):
    name: str
    type: TypeFamily | None = None
    native_type: str | None = None
    """The engine's reflected type name (case-insensitive, for example ``VARCHAR(100)``)."""
    nullable: bool | None = None
    primary_key: bool | None = None
    max_length: int | None = None
    has_default: bool | None = None
    default: str | None = None
    """Expected server default, compared after removing quotes and ``::casts``."""


class ForeignKeyExpectation(_Model):
    columns: tuple[str, ...]
    referred_table: str
    referred_columns: tuple[str, ...]
    on_delete: str | None = None
    """``CASCADE``, ``RESTRICT``, ``SET NULL`` ... compared case-insensitively when set."""


class IndexExpectation(_Model):
    columns: tuple[str, ...]
    unique: bool | None = None
    name: str | None = None


class TableExpectation(_Model):
    name: str
    columns: tuple[ColumnExpectation, ...] = ()
    exact_columns: bool = False
    """Also fail on columns that the expectation does not list."""
    primary_key: tuple[str, ...] | None = None
    foreign_keys: tuple[ForeignKeyExpectation, ...] = ()
    unique_constraints: tuple[tuple[str, ...], ...] = ()
    indexes: tuple[IndexExpectation, ...] = ()


class SchemaExpectation(_Model):
    tables: tuple[TableExpectation, ...] = ()
    views: tuple[str, ...] = ()
    schema_name: str | None = Field(default=None, alias="schema")
    exact_tables: bool = False
    """Also fail on tables that the expectation does not list."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    def table(self, name: str) -> TableExpectation | None:
        return next((t for t in self.tables if t.name == name), None)


class SchemaDifference(_Model):
    kind: str
    subject: str
    expected: str | None = None
    actual: str | None = None

    def __str__(self) -> str:
        detail = ""
        if self.expected is not None or self.actual is not None:
            detail = f": expected {self.expected}, actual {self.actual}"
        return f"[{self.kind}] {self.subject}{detail}"


# --- reflection --------------------------------------------------------------------------


def type_family(column_type: Any) -> TypeFamily:
    if isinstance(column_type, sat.Boolean):
        return "boolean"
    if isinstance(column_type, sat.Integer):
        return "integer"
    if isinstance(column_type, sat.Float):
        return "float"
    if isinstance(column_type, sat.Numeric):
        return "numeric"
    if isinstance(column_type, sat.String):
        return "string"
    if isinstance(column_type, sat.DateTime):
        return "datetime"
    if isinstance(column_type, sat.Date):
        return "date"
    if isinstance(column_type, sat.Time):
        return "time"
    if isinstance(column_type, (sat.LargeBinary, sat.BINARY, sat.VARBINARY)):
        return "binary"
    if isinstance(column_type, sat.JSON):
        return "json"
    if isinstance(column_type, sat.Uuid):
        return "uuid"
    return "other"


def _normalize_default(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"::[A-Za-z_ ]+(\([^)]*\))?(\[\])?", "", value.strip())
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return text.lower()


def _connection_of(source: DatabaseSession | Connection) -> Connection:
    return source if isinstance(source, Connection) else source.connection


def snapshot_schema(source: DatabaseSession | Connection, *, schema: str | None = None) -> SchemaExpectation:
    """Reflect tables, views, columns, keys, constraints and indexes into a fully populated expectation."""
    inspector = inspect(_connection_of(source))
    tables = []
    for name in sorted(inspector.get_table_names(schema=schema)):
        pk_info = inspector.get_pk_constraint(name, schema=schema)
        pk = tuple(pk_info.get("constrained_columns") or ())
        columns = []
        for col in inspector.get_columns(name, schema=schema):
            native = str(col["type"])
            columns.append(
                ColumnExpectation(
                    name=col["name"],
                    type=type_family(col["type"]),
                    native_type=native,
                    nullable=bool(col["nullable"]),
                    primary_key=col["name"] in pk,
                    max_length=getattr(col["type"], "length", None),
                    has_default=col.get("default") is not None,
                    default=_normalize_default(col.get("default")),
                )
            )
        fks = tuple(
            ForeignKeyExpectation(
                columns=tuple(fk["constrained_columns"]),
                referred_table=fk["referred_table"],
                referred_columns=tuple(fk["referred_columns"]),
                on_delete=(fk.get("options") or {}).get("ondelete"),
            )
            for fk in inspector.get_foreign_keys(name, schema=schema)
        )
        uniques = tuple(
            tuple(c for c in uc["column_names"] if c is not None)
            for uc in inspector.get_unique_constraints(name, schema=schema)
        )
        indexes = tuple(
            IndexExpectation(
                columns=tuple(c for c in ix["column_names"] if c is not None),
                unique=bool(ix.get("unique")),
                name=ix.get("name"),
            )
            for ix in inspector.get_indexes(name, schema=schema)
            if ix["column_names"] and all(ix["column_names"])  # skip expression indexes
        )
        tables.append(
            TableExpectation(
                name=name,
                columns=tuple(columns),
                exact_columns=True,
                primary_key=pk,
                foreign_keys=fks,
                unique_constraints=uniques,
                indexes=indexes,
            )
        )
    views = tuple(sorted(inspector.get_view_names(schema=schema)))
    return SchemaExpectation(tables=tuple(tables), views=views, schema=schema, exact_tables=True)


# --- comparison -----------------------------------------------------------------------------


def compare_schema(expected: SchemaExpectation, actual: SchemaExpectation) -> list[SchemaDifference]:
    """Differences between an expectation and a reflected ``actual`` snapshot (empty means it holds)."""
    diffs: list[SchemaDifference] = []
    actual_tables = {t.name: t for t in actual.tables}
    for want in expected.tables:
        have = actual_tables.get(want.name)
        if have is None:
            diffs.append(SchemaDifference(kind="missing_table", subject=want.name))
            continue
        diffs.extend(_compare_table(want, have))
    if expected.exact_tables:
        known = {t.name for t in expected.tables}
        diffs.extend(SchemaDifference(kind="unexpected_table", subject=n) for n in sorted(actual_tables) if n not in known)
    diffs.extend(
        SchemaDifference(kind="missing_view", subject=view) for view in expected.views if view not in actual.views
    )
    return diffs


def _compare_table(want: TableExpectation, have: TableExpectation) -> list[SchemaDifference]:
    diffs: list[SchemaDifference] = []
    name = want.name
    have_columns = {c.name: c for c in have.columns}
    for col in want.columns:
        actual = have_columns.get(col.name)
        subject = f"{name}.{col.name}"
        if actual is None:
            diffs.append(SchemaDifference(kind="missing_column", subject=subject))
            continue
        diffs.extend(_compare_column(subject, col, actual))
    if want.exact_columns:
        known = {c.name for c in want.columns}
        diffs.extend(
            SchemaDifference(kind="unexpected_column", subject=f"{name}.{c}") for c in have_columns if c not in known
        )
    if want.primary_key is not None and tuple(want.primary_key) != tuple(have.primary_key or ()):
        diffs.append(
            SchemaDifference(
                kind="primary_key_mismatch",
                subject=name,
                expected=str(list(want.primary_key)),
                actual=str(list(have.primary_key or ())),
            )
        )
    for fk in want.foreign_keys:
        match = next(
            (
                h
                for h in have.foreign_keys
                if h.columns == fk.columns
                and h.referred_table == fk.referred_table
                and h.referred_columns == fk.referred_columns
            ),
            None,
        )
        label = f"{name}({', '.join(fk.columns)}) -> {fk.referred_table}({', '.join(fk.referred_columns)})"
        if match is None:
            diffs.append(SchemaDifference(kind="missing_foreign_key", subject=label))
        elif fk.on_delete is not None and (match.on_delete or "NO ACTION").upper() != fk.on_delete.upper():
            diffs.append(
                SchemaDifference(
                    kind="foreign_key_on_delete_mismatch",
                    subject=label,
                    expected=fk.on_delete.upper(),
                    actual=(match.on_delete or "NO ACTION").upper(),
                )
            )
    have_uniques = {tuple(u) for u in have.unique_constraints} | {
        tuple(i.columns) for i in have.indexes if i.unique
    }
    diffs.extend(
        SchemaDifference(kind="missing_unique_constraint", subject=f"{name}({', '.join(u)})")
        for u in want.unique_constraints
        if tuple(u) not in have_uniques
    )
    for ix in want.indexes:
        found = [h for h in have.indexes if h.columns == ix.columns and (ix.name is None or h.name == ix.name)]
        subject = f"{name}({', '.join(ix.columns)})"
        if not found:
            diffs.append(SchemaDifference(kind="missing_index", subject=subject))
        elif ix.unique is not None and not any(bool(h.unique) == ix.unique for h in found):
            diffs.append(
                SchemaDifference(kind="index_uniqueness_mismatch", subject=subject, expected=str(ix.unique), actual=str(not ix.unique))
            )
    return diffs


def _compare_column(subject: str, want: ColumnExpectation, have: ColumnExpectation) -> list[SchemaDifference]:
    diffs: list[SchemaDifference] = []

    def check(kind: str, expected: Any, actual: Any) -> None:
        if expected is not None and expected != actual:
            diffs.append(SchemaDifference(kind=kind, subject=subject, expected=str(expected), actual=str(actual)))

    check("type_mismatch", want.type, have.type)
    if want.native_type is not None and want.native_type.lower() != (have.native_type or "").lower():
        diffs.append(
            SchemaDifference(kind="native_type_mismatch", subject=subject, expected=want.native_type, actual=have.native_type)
        )
    check("nullable_mismatch", want.nullable, have.nullable)
    check("primary_key_mismatch", want.primary_key, have.primary_key)
    check("max_length_mismatch", want.max_length, have.max_length)
    check("has_default_mismatch", want.has_default, have.has_default)
    if want.default is not None:
        check("default_mismatch", _normalize_default(want.default), have.default)
    return diffs


# --- public checks ----------------------------------------------------------------------------


def validate_schema(source: DatabaseSession | Connection, expected: SchemaExpectation) -> list[SchemaDifference]:
    """Reflect the database and return the differences from ``expected``."""
    return compare_schema(expected, snapshot_schema(source, schema=expected.schema_name))


def assert_schema(source: DatabaseSession | Connection, expected: SchemaExpectation) -> None:
    """Raise :class:`SchemaValidationError` listing every difference from ``expected``."""
    diffs = validate_schema(source, expected)
    if diffs:
        shown = "\n".join(f"  - {d}" for d in diffs[:_MAX_REPORTED])
        more = f"\n  ... and {len(diffs) - _MAX_REPORTED} more" if len(diffs) > _MAX_REPORTED else ""
        raise SchemaValidationError(f"schema differs from the expectation ({len(diffs)} difference(s)):\n{shown}{more}")


def assert_table_exists(source: DatabaseSession | Connection, name: str, *, schema: str | None = None) -> None:
    if name not in inspect(_connection_of(source)).get_table_names(schema=schema):
        raise SchemaValidationError(f"table {name!r} does not exist")


def assert_view_exists(source: DatabaseSession | Connection, name: str, *, schema: str | None = None) -> None:
    if name not in inspect(_connection_of(source)).get_view_names(schema=schema):
        raise SchemaValidationError(f"view {name!r} does not exist")
