"""Database exceptions, all classified through the shared-core ``ErrorCategory``.

Driver exceptions are wrapped, never re-raised as they are: a driver message can
contain bound values (PostgreSQL ``DETAIL: Key (email)=(...)``) and SQLAlchemy
errors embed the statement. The wrapper keeps the first message line, the
exception type and the SQLSTATE, drops everything else, and stores the original
exception on ``original`` so no cause is lost. It is deliberately *not* chained
with ``raise ... from``, because pytest would print the chained driver message
into the report.

Nothing here is retryable. Deadlocks and serialization failures surface as
:class:`QueryExecutionError` with ``kind="deadlock"`` or ``"serialization"``:
repeating a transaction blindly can hide a real concurrency defect or repeat a
partially applied write, so the test decides.
"""

from __future__ import annotations

from typing import Any, Literal

from complete_automation.core.exceptions import ErrorCategory, FrameworkError
from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor

ErrorKind = Literal[
    "connection",
    "timeout",
    "integrity",
    "deadlock",
    "serialization",
    "syntax",
    "permission",
    "data",
    "unsupported",
    "other",
]

_MESSAGE_LIMIT = 300


class DatabaseError(FrameworkError):
    """Base class for database module errors."""

    original: BaseException | None = None
    """The driver or SQLAlchemy exception this error wraps, when there is one."""


class DatabaseConfigurationError(DatabaseError):
    """Settings are invalid, inconsistent or request something the adapter cannot do."""

    category = ErrorCategory.CONFIGURATION


class UnsafeOperationError(DatabaseError):
    """A statement was refused by the operation policy (read-only mode, DDL or destructive guard)."""

    category = ErrorCategory.CONFIGURATION


class UnsupportedOperationError(DatabaseError):
    """The adapter or database engine does not support the requested operation."""

    category = ErrorCategory.CONFIGURATION


class DatabaseConnectionError(DatabaseError):
    """A connection could not be opened or was lost."""

    category = ErrorCategory.INFRASTRUCTURE


class QueryExecutionError(DatabaseError):
    """A statement failed in the database (see ``kind``). The driver error is on ``original``."""

    category = ErrorCategory.UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = "other",
        original: BaseException | None = None,
        sqlstate: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.original = original
        self.sqlstate = sqlstate
        self.operation = operation
        self.category = _CATEGORY_BY_KIND.get(kind, ErrorCategory.UNKNOWN)


class DatabaseIntegrityError(QueryExecutionError):
    """A constraint (unique, foreign key, not null, check) rejected the statement."""


class QueryTimeoutError(QueryExecutionError):
    """The statement or the wait for a connection or lock exceeded its timeout."""


class ResultTooLargeError(DatabaseError):
    """A query returned more rows than ``max_rows``; stream it with ``iterate`` instead."""

    category = ErrorCategory.TEST_DATA


class TransactionError(DatabaseError):
    """Transaction control was used incorrectly, or a commit or rollback failed."""

    category = ErrorCategory.INFRASTRUCTURE


class CleanupError(DatabaseError):
    """Cleanup (rollback, connection release, sandbox removal) failed after the test body ran."""

    category = ErrorCategory.INFRASTRUCTURE


class SchemaValidationError(DatabaseError, AssertionError):
    """The database schema differs from the declared expectation."""

    category = ErrorCategory.ASSERTION


class DataAssertionError(DatabaseError, AssertionError):
    """A data check failed: the database content is not what the test expected."""

    category = ErrorCategory.ASSERTION


class ReconciliationError(DatabaseError):
    """Reconciliation could not run (for example, inputs that are not ordered by key)."""

    category = ErrorCategory.TEST_DATA


_CATEGORY_BY_KIND: dict[str, ErrorCategory] = {
    "connection": ErrorCategory.INFRASTRUCTURE,
    "timeout": ErrorCategory.TIMEOUT,
    "integrity": ErrorCategory.ASSERTION,
    "deadlock": ErrorCategory.INFRASTRUCTURE,
    "serialization": ErrorCategory.INFRASTRUCTURE,
    "syntax": ErrorCategory.UNKNOWN,
    "permission": ErrorCategory.CONFIGURATION,
    "data": ErrorCategory.TEST_DATA,
    "unsupported": ErrorCategory.CONFIGURATION,
    "other": ErrorCategory.UNKNOWN,
}


def safe_driver_message(error: BaseException, redactor: Redactor = DEFAULT_REDACTOR) -> str:
    """First line of the driver message, redacted and truncated; never the SQL or parameters."""
    orig = getattr(error, "orig", None) or error
    diag = getattr(orig, "diag", None)
    primary = getattr(diag, "message_primary", None)  # psycopg: without DETAIL/HINT
    text = primary if isinstance(primary, str) and primary else str(orig)
    first_line = (text.strip().splitlines() or [""])[0]
    return redactor.redact_text(first_line)[:_MESSAGE_LIMIT]


def sqlstate_of(error: BaseException) -> str | None:
    orig: Any = getattr(error, "orig", None) or error
    state = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    return state if isinstance(state, str) and state else None
