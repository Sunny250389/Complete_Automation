"""Safe handling of dynamic identifiers (table, column and schema names).

Identifiers cannot be bound parameters. The validation helpers never put a name
into SQL text: they check it against a strict pattern (and an optional
allowlist) and then build SQLAlchemy expressions, which quote it correctly.
Values in conditions are always bound.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, ColumnElement, Date, DateTime, Numeric, TableClause, column, table
from sqlalchemy.sql.type_api import TypeEngine

from complete_automation.database.errors import UnsafeOperationError
from complete_automation.database.settings import IDENTIFIER


def safe_identifier(name: str, *, allowed: Collection[str] | None = None, what: str = "identifier") -> str:
    """Return ``name`` if it is a plain identifier (and in ``allowed``, when given).

    Raises:
        UnsafeOperationError: the name has characters outside ``[A-Za-z0-9_$]``,
            starts with a digit, is too long, or is not in the allowlist.
    """
    if not isinstance(name, str) or not IDENTIFIER.match(name):
        raise UnsafeOperationError(f"refused: {what} {name!r} is not a plain identifier")
    if allowed is not None and name not in allowed:
        raise UnsafeOperationError(f"refused: {what} {name!r} is not in the allowed names")
    return name


def table_ref(name: str, schema: str | None = None, *, allowed: Collection[str] | None = None) -> TableClause:
    """A quoted table reference built with SQLAlchemy, never string formatting."""
    return table(
        safe_identifier(name, allowed=allowed, what="table name"),
        schema=safe_identifier(schema, what="schema name") if schema else None,
    )


def table_with_columns(
    name: str, columns: Sequence[str], schema: str | None = None, *, allowed: Collection[str] | None = None
) -> TableClause:
    """A table reference that declares ``columns``, so joins can use ``t.c.<column>``."""
    return table(
        safe_identifier(name, allowed=allowed, what="table name"),
        *[column(safe_identifier(c, what="column name")) for c in columns],
        schema=safe_identifier(schema, what="schema name") if schema else None,
    )


def column_names(columns: str | Sequence[str]) -> list[str]:
    names = [columns] if isinstance(columns, str) else list(columns)
    if not names:
        raise UnsafeOperationError("refused: at least one column is required")
    return [safe_identifier(name, what="column name") for name in names]


def _type_for(value: Any) -> TypeEngine[Any] | None:
    """Give datetime-like and decimal values a type so SQLAlchemy binds them portably."""
    if isinstance(value, dt.datetime):
        return DateTime()
    if isinstance(value, dt.date):
        return Date()
    if isinstance(value, Decimal):
        return Numeric()
    if isinstance(value, bool):
        return Boolean()
    return None


def typed_column(name: str, sample: Any = None) -> ColumnElement[Any]:
    return column(safe_identifier(name, what="column name"), _type_for(sample))


def conditions(where: Mapping[str, Any] | None) -> list[ColumnElement[bool]]:
    """``{"status": "open", "deleted_at": None, "id": [1, 2]}`` as bound equality / IS NULL / IN."""
    clauses: list[ColumnElement[bool]] = []
    for name, value in (where or {}).items():
        col = typed_column(name, value if not isinstance(value, (list, tuple, set, frozenset)) else None)
        if value is None:
            clauses.append(col.is_(None))
        elif isinstance(value, (list, tuple, set, frozenset)):
            clauses.append(col.in_(list(value)))
        else:
            clauses.append(col == value)
    return clauses
