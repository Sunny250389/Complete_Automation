"""One connection with an executor and explicit transaction control.

A :class:`DatabaseSession` is what tests and fixtures hold. It is not shared
between tests and not thread-safe (it wraps one DBAPI connection). Leaving the
``with`` block rolls back anything uncommitted and always releases the
connection, also after assertion failures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any

from sqlalchemy import MetaData
from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import Connection

from complete_automation.core.log import get_logger
from complete_automation.database.classification import wrap_error
from complete_automation.database.errors import CleanupError, DatabaseConfigurationError
from complete_automation.database.executor import Params, QueryResult, QuerySummary, SqlExecutor, Statement
from complete_automation.database.transactions import TransactionManager, TransactionOutcome

if TYPE_CHECKING:
    from complete_automation.database.engine import EngineManager

_logger = get_logger("database")


class DatabaseSession:
    def __init__(
        self,
        manager: EngineManager,
        *,
        commit_on_success: bool = False,
        isolation_level: str | None = None,
        on_query: Callable[[QuerySummary], None] | None = None,
    ) -> None:
        self.manager = manager
        self.commit_on_success = commit_on_success
        self.connection: Connection = manager.connect()
        self._closed = False
        try:
            if isolation_level is not None:
                level = isolation_level.upper()
                if level not in manager.adapter.capabilities.isolation_levels:
                    raise DatabaseConfigurationError(
                        f"isolation_level {isolation_level!r} is not supported by {manager.dialect}"
                    )
                try:
                    self.connection.execution_options(isolation_level=level)
                except sa_exc.SQLAlchemyError as error:
                    raise wrap_error(error, manager.adapter, operation="set isolation level") from None
            self.executor = SqlExecutor(
                self.connection, manager, max_rows=manager.settings.max_rows, on_query=on_query
            )
            self.transactions = TransactionManager(self.connection, manager.adapter)
        except BaseException:
            self.connection.close()
            raise

    # --- statements (delegated to the executor) ----------------------------

    def query(self, statement: Statement, params: Params = None, *, max_rows: int | None = None) -> QueryResult:
        return self.executor.query(statement, params, max_rows=max_rows)

    def fetch_all(self, statement: Statement, params: Params = None) -> list[dict[str, Any]]:
        return self.executor.fetch_all(statement, params)

    def fetch_one(self, statement: Statement, params: Params = None) -> dict[str, Any] | None:
        return self.executor.fetch_one(statement, params)

    def scalar(self, statement: Statement, params: Params = None) -> Any:
        return self.executor.scalar(statement, params)

    def iterate(self, statement: Statement, params: Params = None, *, chunk_size: int = 1000) -> Iterator[dict[str, Any]]:
        return self.executor.iterate(statement, params, chunk_size=chunk_size)

    def execute(
        self, statement: Statement, params: Params = None, *, expect_rowcount: int | None = None
    ) -> QueryResult:
        return self.executor.execute(statement, params, expect_rowcount=expect_rowcount)

    def execute_many(self, statement: Statement, rows: Sequence[Mapping[str, Any]]) -> QueryResult:
        return self.executor.execute_many(statement, rows)

    def create_all(self, metadata: MetaData) -> None:
        self.executor.create_all(metadata)

    def drop_all(self, metadata: MetaData) -> None:
        self.executor.drop_all(metadata)

    # --- transactions ------------------------------------------------------------

    def commit(self) -> None:
        self.transactions.commit()

    def rollback(self) -> None:
        self.transactions.rollback()

    def savepoint(self) -> AbstractContextManager[None]:
        return self.transactions.savepoint()

    def rolled_back_savepoint(self) -> AbstractContextManager[None]:
        return self.transactions.rolled_back_savepoint()

    def transaction(self, *, commit: bool = False) -> AbstractContextManager[None]:
        return self.transactions.transaction(commit=commit)

    @property
    def history(self) -> list[QuerySummary]:
        """Summaries of the most recent statements (fingerprints only, never parameters)."""
        return list(self.executor.history)

    @property
    def transaction_outcomes(self) -> list[TransactionOutcome]:
        return list(self.transactions.outcomes)

    # --- lifecycle -------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Roll back uncommitted work and release the connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        problems: list[str] = []
        try:
            self.transactions.close()
        except CleanupError as error:
            problems.append(str(error))
        try:
            self.connection.close()
        except sa_exc.SQLAlchemyError as error:
            problems.append(f"connection close failed [{type(error).__name__}]")
        if problems:
            raise CleanupError("; ".join(problems))

    def __enter__(self) -> DatabaseSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None and self.commit_on_success and self.transactions.active:
                self.commit()
        finally:
            try:
                self.close()
            except CleanupError:
                if exc_type is None:
                    raise
                _logger.warning("database session cleanup failed after a test error")
