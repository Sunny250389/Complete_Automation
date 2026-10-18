"""Safe SQL execution on one connection.

- Values are always bound parameters. Pass a mapping for named binds
  (``:name``) or a sequence for positional ``?`` placeholders. A literal colon
  in a SQL string needs escaping as ``\\:`` (SQLAlchemy ``text()`` rule); a literal
  ``?`` (for example the PostgreSQL JSONB operator) must be avoided with
  positional parameters, so use named parameters there.
- Identifiers (tables, columns) are *not* parameters. Build dynamic ones with
  SQLAlchemy expressions (``table("t")``, ``column("c")``), or check them with
  :func:`~complete_automation.database.validations.identifiers.safe_identifier`.
- Every statement passes the :class:`OperationPolicy` before it runs.
- Nothing commits. Transaction control is in :mod:`transactions`.
- Failures are wrapped by :func:`wrap_error`: no SQL text and no bound values in
  messages, original exception on ``.original``.
- Writes are never retried here.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import MetaData, text
from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import Connection
from sqlalchemy.sql import ClauseElement, Executable

from complete_automation.core.log import get_logger
from complete_automation.database.classification import wrap_error
from complete_automation.database.errors import (
    DataAssertionError,
    DatabaseError,
    ResultTooLargeError,
)
from complete_automation.database.sqltext import (
    OperationKind,
    StatementInfo,
    classify,
    fingerprint,
    positional_to_named,
)

if TYPE_CHECKING:
    from complete_automation.database.engine import EngineManager

Params = Mapping[str, Any] | Sequence[Any] | None
Statement = str | Executable
QuerySummary = dict[str, Any]

_logger = get_logger("database")
HISTORY_LIMIT = 50


@dataclass(frozen=True)
class QueryResult:
    """Outcome of one statement. ``rows`` are dicts keyed by column name."""

    operation: OperationKind
    verb: str
    statement: str
    """Fingerprint: comments removed, inline literals replaced by ``?``. Safe to report."""
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    rowcount: int | None
    """Rows affected by a write; ``None`` for reads (and when the driver cannot tell)."""
    duration_ms: float
    dialect: str

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def first(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    def one(self) -> dict[str, Any]:
        """The only row; raises :class:`DataAssertionError` for zero or several."""
        if len(self.rows) != 1:
            raise DataAssertionError(f"expected exactly one row, got {len(self.rows)}")
        return self.rows[0]

    def scalar(self) -> Any:
        """First column of the only row."""
        row = self.one()
        return next(iter(row.values()))

    def column(self, name: str) -> list[Any]:
        if name not in self.columns:
            raise DataAssertionError(f"result has no column {name!r} (columns: {', '.join(self.columns)})")
        return [row[name] for row in self.rows]

    def summary(self) -> QuerySummary:
        return {
            "outcome": "ok",
            "operation": self.operation.value,
            "verb": self.verb,
            "statement": self.statement,
            "rows": self.row_count,
            "rowcount": self.rowcount,
            "duration_ms": round(self.duration_ms, 3),
            "dialect": self.dialect,
        }


@dataclass
class SqlExecutor:
    connection: Connection
    manager: EngineManager
    max_rows: int
    on_query: Callable[[QuerySummary], None] | None = None
    history: deque[QuerySummary] = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))

    # --- reads -------------------------------------------------------------

    def query(self, statement: Statement, params: Params = None, *, max_rows: int | None = None) -> QueryResult:
        """Run a read statement and return all rows (at most ``max_rows``, else :class:`ResultTooLargeError`)."""
        limit = max_rows or self.max_rows
        prepared = self._prepare(statement, params, expect=(OperationKind.READ,))
        started = time.perf_counter()
        try:
            result = self.connection.execute(prepared.clause, prepared.params)
            columns = tuple(result.keys()) if result.returns_rows else ()
            raw = result.fetchmany(limit + 1) if result.returns_rows else []
            result.close()
        except sa_exc.SQLAlchemyError as error:
            raise self._failed(prepared, error, started) from None
        if len(raw) > limit:
            raise ResultTooLargeError(
                f"query returned more than {limit} rows; use iterate() to stream it or raise max_rows"
            )
        rows = tuple(dict(zip(columns, row, strict=True)) for row in raw)
        return self._finish(prepared, started, columns, rows, None)

    def fetch_all(self, statement: Statement, params: Params = None) -> list[dict[str, Any]]:
        return list(self.query(statement, params).rows)

    def fetch_one(self, statement: Statement, params: Params = None) -> dict[str, Any] | None:
        """The first row or ``None``. Use ``query(...).one()`` to require exactly one."""
        return self.query(statement, params).first()

    def scalar(self, statement: Statement, params: Params = None) -> Any:
        """The first column of the single result row (aggregates, ``count(*)``, ...)."""
        return self.query(statement, params).scalar()

    def iterate(
        self, statement: Statement, params: Params = None, *, chunk_size: int = 1000
    ) -> Iterator[dict[str, Any]]:
        """Stream a read statement row by row, fetching ``chunk_size`` rows at a time.

        The statement is checked when iteration starts. Close the iterator (or
        exhaust it) before running another statement on this connection.
        """
        prepared = self._prepare(statement, params, expect=(OperationKind.READ,))
        started = time.perf_counter()
        count = 0
        try:
            result = self.connection.execute(
                prepared.clause,
                prepared.params,
                execution_options={"stream_results": True, "max_row_buffer": chunk_size},
            )
            columns = tuple(result.keys())
            try:
                while chunk := result.fetchmany(chunk_size):
                    for row in chunk:
                        count += 1
                        yield dict(zip(columns, row, strict=True))
            finally:
                result.close()
        except sa_exc.SQLAlchemyError as error:
            raise self._failed(prepared, error, started) from None
        self._finish(prepared, started, columns, (), None, streamed_rows=count)

    # --- writes ----------------------------------------------------------------

    def execute(
        self, statement: Statement, params: Params = None, *, expect_rowcount: int | None = None
    ) -> QueryResult:
        """Run a statement that changes data (INSERT/UPDATE/DELETE, DDL when authorized).

        The change stays inside the current transaction until the caller commits.
        ``expect_rowcount`` raises :class:`DataAssertionError` when the number of
        affected rows differs; the statement has then already run, so roll back.
        """
        prepared = self._prepare(statement, params)
        started = time.perf_counter()
        try:
            result = self.connection.execute(prepared.clause, prepared.params)
            columns = tuple(result.keys()) if result.returns_rows else ()
            rows = tuple(dict(zip(columns, r, strict=True)) for r in result.fetchall()) if result.returns_rows else ()
            if result.returns_rows:
                # DML with RETURNING yields one row per affected row. Drivers disagree on rowcount here
                # (SQLite reports it only after the rows are consumed, psycopg not at all), so derive it.
                rowcount = len(rows) if prepared.info.kind in (OperationKind.WRITE, OperationKind.DESTRUCTIVE) else None
            else:
                rowcount = result.rowcount if result.rowcount >= 0 else None
            result.close()
        except sa_exc.SQLAlchemyError as error:
            raise self._failed(prepared, error, started) from None
        outcome = self._finish(prepared, started, columns, rows, rowcount)
        if expect_rowcount is not None and rowcount != expect_rowcount:
            raise DataAssertionError(
                f"{prepared.info.verb} affected {rowcount} row(s), expected {expect_rowcount}"
            )
        return outcome

    def execute_many(self, statement: Statement, rows: Sequence[Mapping[str, Any]]) -> QueryResult:
        """Run one write statement once per parameter set (batch insert/update)."""
        if not rows:
            raise DatabaseError("execute_many needs at least one parameter set")
        prepared = self._prepare(statement, None)
        if prepared.info.kind is not OperationKind.WRITE:
            raise DatabaseError("execute_many is for INSERT/UPDATE/DELETE statements")
        started = time.perf_counter()
        try:
            result = self.connection.execute(prepared.clause, list(rows))
            rowcount = result.rowcount
            result.close()
        except sa_exc.SQLAlchemyError as error:
            raise self._failed(prepared, error, started) from None
        return self._finish(prepared, started, (), (), rowcount)

    def create_all(self, metadata: MetaData) -> None:
        """``CREATE TABLE`` for every table in ``metadata`` that does not exist. Needs ``allow_ddl``."""
        self._authorize(StatementInfo(OperationKind.DDL, "CREATE", 1))
        self._run_ddl(lambda: metadata.create_all(self.connection), "CREATE TABLE")

    def drop_all(self, metadata: MetaData) -> None:
        """``DROP TABLE`` for every table in ``metadata``. Needs ``allow_destructive``."""
        self._authorize(StatementInfo(OperationKind.DESTRUCTIVE, "DROP", 1))
        self._run_ddl(lambda: metadata.drop_all(self.connection), "DROP TABLE")

    # --- internals -------------------------------------------------------------

    def _run_ddl(self, action: Callable[[], None], label: str) -> None:
        started = time.perf_counter()
        try:
            action()
        except sa_exc.SQLAlchemyError as error:
            summary = self._failure_summary(OperationKind.DDL, label.split()[0], label, error, started)
            self._record(summary)
            raise wrap_error(error, self.manager.adapter, operation=label, redactor=self.manager.redactor) from None
        kind = OperationKind.DESTRUCTIVE if label.startswith("DROP") else OperationKind.DDL
        self._record(
            {
                "outcome": "ok", "operation": kind.value, "verb": label.split()[0], "statement": label,
                "rows": 0, "rowcount": None, "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "dialect": self.manager.dialect,
            }
        )  # fmt: skip

    def _authorize(self, info: StatementInfo) -> None:
        verified = None
        if self.manager.policy.needs_identity_check(info) and not self.manager.policy.read_only:
            verified = self.manager.verified_database(self.connection)
        self.manager.policy.check(info, verified_database=verified)

    def _prepare(
        self, statement: Statement, params: Params, *, expect: tuple[OperationKind, ...] | None = None
    ) -> _Prepared:
        if isinstance(statement, str):
            sql = statement
            if params is not None and not isinstance(params, Mapping):
                if isinstance(params, (str, bytes)):
                    raise DatabaseError("params must be a mapping or a sequence, not a string")
                sql, bound = positional_to_named(sql, list(params))
            else:
                bound = dict(params) if params else {}
            clause: Executable = text(sql)
        else:
            if params is not None and not isinstance(params, Mapping):
                raise DatabaseError("positional parameters are only supported for SQL strings")
            sql = str(cast(ClauseElement, statement).compile(dialect=self.connection.dialect))
            bound = dict(params) if params else {}
            clause = statement
        info = classify(sql)
        if expect is not None and info.kind not in expect:
            raise DatabaseError(
                f"{info.verb or 'statement'} is a {info.kind.value} operation: use execute() instead of a read method"
            )
        self._authorize(info)
        return _Prepared(clause, bound, info, fingerprint(sql))

    def _finish(
        self,
        prepared: _Prepared,
        started: float,
        columns: tuple[str, ...],
        rows: tuple[dict[str, Any], ...],
        rowcount: int | None,
        *,
        streamed_rows: int | None = None,
    ) -> QueryResult:
        duration = (time.perf_counter() - started) * 1000
        result = QueryResult(
            operation=prepared.info.kind,
            verb=prepared.info.verb,
            statement=prepared.statement,
            columns=columns,
            rows=rows,
            rowcount=rowcount if rowcount is None or rowcount >= 0 else None,
            duration_ms=duration,
            dialect=self.manager.dialect,
        )
        summary = result.summary()
        if streamed_rows is not None:
            summary["rows"] = streamed_rows
        self._record(summary)
        return result

    def _failed(self, prepared: _Prepared, error: sa_exc.SQLAlchemyError, started: float) -> DatabaseError:
        self._record(self._failure_summary(prepared.info.kind, prepared.info.verb, prepared.statement, error, started))
        return wrap_error(error, self.manager.adapter, operation=prepared.info.verb or "statement", redactor=self.manager.redactor)

    def _failure_summary(
        self, kind: OperationKind, verb: str, statement: str, error: sa_exc.SQLAlchemyError, started: float
    ) -> QuerySummary:
        wrapped = wrap_error(error, self.manager.adapter, redactor=self.manager.redactor)
        return {
            "outcome": "error",
            "operation": kind.value,
            "verb": verb,
            "statement": statement,
            "error_kind": getattr(wrapped, "kind", "connection"),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "dialect": self.manager.dialect,
        }

    def _record(self, summary: QuerySummary) -> None:
        self.history.append(summary)
        _logger.debug(
            "sql %s", summary["outcome"],
            extra={k: summary.get(k) for k in ("operation", "verb", "duration_ms", "rows", "rowcount", "dialect")},
        )  # fmt: skip
        if self.on_query is not None:
            self.on_query(summary)


@dataclass(frozen=True)
class _Prepared:
    clause: Executable
    params: dict[str, Any]
    info: StatementInfo
    statement: str
