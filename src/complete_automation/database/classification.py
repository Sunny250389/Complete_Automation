"""Turn SQLAlchemy and driver exceptions into the module's classified errors."""

from __future__ import annotations

from sqlalchemy import exc as sa_exc

from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor
from complete_automation.database.adapters.base import DatabaseAdapter
from complete_automation.database.errors import (
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    ErrorKind,
    QueryExecutionError,
    QueryTimeoutError,
    safe_driver_message,
    sqlstate_of,
)


def error_kind(error: sa_exc.SQLAlchemyError, adapter: DatabaseAdapter | None) -> ErrorKind:
    if isinstance(error, sa_exc.TimeoutError):  # waiting for a pooled connection
        return "timeout"
    if isinstance(error, sa_exc.DBAPIError):
        if adapter is not None and (kind := adapter.classify(error)) is not None:
            return kind
        if error.connection_invalidated or isinstance(error, sa_exc.InterfaceError):
            return "connection"
        if isinstance(error, sa_exc.IntegrityError):
            return "integrity"
        if isinstance(error, sa_exc.DataError):
            return "data"
        if isinstance(error, sa_exc.ProgrammingError):
            return "syntax"
        if isinstance(error, sa_exc.OperationalError):
            return "connection" if _looks_like_connection_failure(error) else "other"
        if isinstance(error, sa_exc.NotSupportedError):
            return "unsupported"
    return "other"


def _looks_like_connection_failure(error: sa_exc.DBAPIError) -> bool:
    text = str(getattr(error, "orig", error)).lower()
    return any(
        hint in text
        for hint in ("could not connect", "connection refused", "connection failed", "timeout expired",
                     "no route to host", "name or service not known", "password authentication",
                     "server closed the connection", "connection to server")
    )  # fmt: skip


def wrap_error(
    error: sa_exc.SQLAlchemyError,
    adapter: DatabaseAdapter | None,
    *,
    operation: str | None = None,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> DatabaseError:
    """Wrap ``error`` without its SQL or bound values; the original stays on ``.original``."""
    kind = error_kind(error, adapter)
    detail = safe_driver_message(error, redactor)
    where = f" during {operation}" if operation else ""
    label = {
        "connection": "database connection failed",
        "timeout": "database operation timed out",
        "integrity": "constraint violation",
        "deadlock": "deadlock detected",
        "serialization": "serialization failure",
        "syntax": "invalid SQL",
        "permission": "permission denied",
        "data": "invalid data",
        "unsupported": "unsupported operation",
        "other": "database error",
    }[kind]
    message = f"{label}{where} [{type(getattr(error, 'orig', error)).__name__}]: {detail}"
    if kind == "connection":
        wrapped: DatabaseError = DatabaseConnectionError(message)
        wrapped.original = error
        return wrapped
    cls = {"integrity": DatabaseIntegrityError, "timeout": QueryTimeoutError}.get(kind, QueryExecutionError)
    return cls(message, kind=kind, original=error, sqlstate=sqlstate_of(error), operation=operation)
