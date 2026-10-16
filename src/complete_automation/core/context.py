"""Run-level and test-level execution metadata.

Contexts are immutable records. The *current* context is tracked with
``contextvars``, so each thread and each asyncio task sees its own value and
parallel tests cannot overwrite each other's metadata. A thread started with
``threading.Thread`` begins with no current context; pass the context
explicitly or use ``contextvars.copy_context().run(...)``. Under pytest-xdist
each worker is a separate process with its own state.

Contexts hold identifiers and plain metadata only, never browsers, sessions,
connections or clients.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_run_id(now: datetime | None = None) -> str:
    """Return ``<UTC timestamp>-<8 hex chars>``, sortable and collision resistant."""
    stamp = (now or utc_now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def new_correlation_id() -> str:
    return secrets.token_hex(8)


class _Context(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _copy(cls, value: Any) -> Any:
        # Copy so later changes to the caller's dict do not leak into the context.
        return dict(value) if isinstance(value, Mapping) else value


class RunContext(_Context):
    """One test run (a pytest session, a CI job, a Kubernetes Job shard)."""

    run_id: str
    environment: str
    started_at: datetime
    finished_at: datetime | None = None
    suite: str | None = None

    @classmethod
    def start(
        cls,
        environment: str,
        *,
        run_id: str | None = None,
        suite: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> RunContext:
        started = utc_now()
        return cls(
            run_id=run_id or new_run_id(started),
            environment=environment,
            started_at=started,
            suite=suite,
            metadata=dict(metadata or {}),
        )

    def finish(self) -> RunContext:
        return self.model_copy(update={"finished_at": utc_now()})


class TestContext(_Context):
    """One test within a run."""

    __test__ = False  # not a pytest test class

    run_id: str
    test_id: str
    name: str
    module: str
    correlation_id: str = Field(default_factory=new_correlation_id)
    started_at: datetime = Field(default_factory=utc_now)


_current_run: ContextVar[RunContext | None] = ContextVar("cauto_run", default=None)
_current_test: ContextVar[TestContext | None] = ContextVar("cauto_test", default=None)


def current_run_context() -> RunContext | None:
    return _current_run.get()


def current_test_context() -> TestContext | None:
    return _current_test.get()


@contextmanager
def use_run_context(context: RunContext) -> Iterator[RunContext]:
    token = _current_run.set(context)
    try:
        yield context
    finally:
        _current_run.reset(token)


def activate_run_context(context: RunContext) -> Token[RunContext | None]:
    """Make ``context`` current until :func:`deactivate_run_context` (for hook pairs)."""
    return _current_run.set(context)


def deactivate_run_context(token: Token[RunContext | None]) -> None:
    try:
        _current_run.reset(token)
    except ValueError:  # token from another context: clear rather than leak the run
        _current_run.set(None)


@contextmanager
def use_test_context(context: TestContext) -> Iterator[TestContext]:
    token = _current_test.set(context)
    try:
        yield context
    finally:
        _current_test.reset(token)
