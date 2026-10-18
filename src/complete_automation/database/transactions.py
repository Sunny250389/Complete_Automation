"""Explicit transaction control on one connection.

Rules this module enforces:

- Nothing commits unless ``commit()`` is called (or ``transaction(commit=True)``
  succeeds). A connection that is closed with work pending rolls it back.
- A failed ``transaction()`` or ``savepoint()`` block rolls back and re-raises
  the original error; a rollback that itself fails raises :class:`CleanupError`
  without hiding the test failure that caused it (the failure stays the
  ``__context__``).
- Deadlocks and serialization failures are never retried here.
- Engines differ: SQLite has one writer and transactional DDL, PostgreSQL has
  concurrent writers, MVCC isolation levels and transactional DDL. Check
  ``adapter.capabilities`` before relying on savepoints or DDL rollback, and do
  not read SQLite behavior as evidence for PostgreSQL concurrency.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import Connection

from complete_automation.core.log import get_logger
from complete_automation.database.adapters import DatabaseAdapter
from complete_automation.database.errors import CleanupError, TransactionError

_logger = get_logger("database")


@dataclass(frozen=True)
class TransactionOutcome:
    scope: Literal["transaction", "savepoint"]
    outcome: Literal["committed", "rolled_back", "failed"]
    duration_ms: float
    error: str | None = None


@dataclass
class TransactionManager:
    connection: Connection
    adapter: DatabaseAdapter
    outcomes: list[TransactionOutcome] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.connection.in_transaction()

    def begin(self) -> None:
        """Start a transaction explicitly. Statements also start one implicitly (autobegin)."""
        if self.active:
            raise TransactionError("a transaction is already active on this connection")
        self.connection.begin()

    def commit(self) -> None:
        started = time.perf_counter()
        try:
            self.connection.commit()
        except sa_exc.SQLAlchemyError as error:
            self._record("transaction", "failed", started, error)
            raise TransactionError(f"commit failed [{type(error).__name__}]") from None
        self._record("transaction", "committed", started)

    def rollback(self) -> None:
        started = time.perf_counter()
        try:
            self.connection.rollback()
        except sa_exc.SQLAlchemyError as error:
            self._record("transaction", "failed", started, error)
            raise CleanupError(f"rollback failed [{type(error).__name__}]") from None
        self._record("transaction", "rolled_back", started)

    @contextmanager
    def transaction(self, *, commit: bool = False) -> Iterator[None]:
        """Run the block in a fresh transaction.

        Commits on success only when ``commit=True``; otherwise rolls back (the
        default: verify, then discard). Any exception rolls back and propagates.

        Every statement starts a transaction implicitly (SQLAlchemy "autobegin"),
        so this raises :class:`TransactionError` when one is already in progress:
        earlier statements cannot be made part of an all-or-nothing block. Call
        ``commit()`` or ``rollback()`` first.
        """
        self.begin()
        try:
            yield
        except BaseException:
            self._rollback_quietly()
            raise
        if commit:
            self.commit()
        else:
            self.rollback()

    @contextmanager
    def savepoint(self) -> Iterator[None]:
        """A nested transaction (``SAVEPOINT``): rolled back on error, released on success."""
        self.adapter.require("savepoints")
        started = time.perf_counter()
        nested = self.connection.begin_nested()
        try:
            yield
        except BaseException:
            if nested.is_active:
                try:
                    nested.rollback()
                except sa_exc.SQLAlchemyError:
                    _logger.warning("savepoint rollback failed")
            self._record("savepoint", "rolled_back", started)
            raise
        nested.commit()
        self._record("savepoint", "committed", started)

    @contextmanager
    def rolled_back_savepoint(self) -> Iterator[None]:
        """Run the block, then *always* discard its changes (inspect, then undo)."""
        self.adapter.require("savepoints")
        started = time.perf_counter()
        nested = self.connection.begin_nested()
        try:
            yield
        finally:
            if nested.is_active:
                nested.rollback()
            self._record("savepoint", "rolled_back", started)

    def close(self) -> None:
        """Roll back anything uncommitted. Safe to call repeatedly."""
        if self.active:
            self.rollback()

    def _rollback_quietly(self) -> None:
        try:
            self.connection.rollback()
        except sa_exc.SQLAlchemyError:
            _logger.warning("rollback after a failed block also failed")
        else:
            self._record("transaction", "rolled_back", time.perf_counter())

    def _record(
        self,
        scope: Literal["transaction", "savepoint"],
        outcome: Literal["committed", "rolled_back", "failed"],
        started: float,
        error: BaseException | None = None,
    ) -> None:
        self.outcomes.append(
            TransactionOutcome(
                scope,
                outcome,
                round((time.perf_counter() - started) * 1000, 3),
                type(error).__name__ if error is not None else None,
            )
        )
