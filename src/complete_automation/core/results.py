"""The framework-wide test result contract.

Every testing module reports outcomes as :class:`TestResult`. Domain detail
travels in typed, optional parts (``metrics``, ``artifacts``) and in
``metadata``, which must be JSON-compatible and non-sensitive.

Status semantics:

- ``passed``: every check passed.
- ``failed``: a check failed (``AssertionError``) or a metric missed its
  threshold; the system under test is at fault.
- ``error``: the test could not run properly (setup, infrastructure,
  configuration, test data, a defect in the test).
- ``skipped``: deliberately not run (includes expected failures, ``xfail``).
- ``incomplete``: ran, but no verdict could be reached (for example an
  evaluator returned an unusable score).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    field_validator,
    model_validator,
)

from complete_automation.core.artifacts import ArtifactRef
from complete_automation.core.exceptions import ErrorCategory

SCHEMA_VERSION: Final = "1.0"

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactRef",
    "AttemptInfo",
    "ErrorCategory",
    "FailureInfo",
    "MetricResult",
    "Status",
    "TestResult",
]


class Status(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    INCOMPLETE = "incomplete"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class MetricResult(_Model):
    """One measured value, for example an evaluation score or a latency."""

    name: str = Field(min_length=1)
    value: float | int | bool | str
    threshold: float | None = None
    passed: bool | None = None
    """The module's verdict for this metric; reporting never recomputes it."""
    unit: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FailureInfo(_Model):
    message: str
    category: ErrorCategory
    exception_type: str | None = None
    details: str | None = None
    """Longer diagnostic text such as a traceback, already redacted."""


class AttemptInfo(_Model):
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)

    @model_validator(mode="after")
    def _within_limit(self) -> AttemptInfo:
        if self.attempt > self.max_attempts:
            raise ValueError("attempt cannot exceed max_attempts")
        return self


class TestResult(_Model):
    """The outcome of one test, independent of the testing discipline."""

    __test__ = False  # not a pytest test class

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    test_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    module: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    """Testing discipline or suite, for example ``api``, ``ui``, ``llm``."""
    status: Status
    started_at: datetime
    finished_at: datetime
    correlation_id: str | None = None
    failure: FailureInfo | None = None
    metrics: tuple[MetricResult, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    attempt: AttemptInfo | None = None
    tags: tuple[str, ...] = ()

    @field_validator("started_at", "finished_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> TestResult:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is earlier than started_at")
        if self.status in (Status.PASSED, Status.SKIPPED) and self.failure is not None:
            raise ValueError(f"a {self.status.value} result cannot carry a failure")
        if self.status in (Status.FAILED, Status.ERROR) and self.failure is None:
            raise ValueError(f"a {self.status.value} result needs failure details")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float:
        return round((self.finished_at - self.started_at).total_seconds(), 6)

    def to_dict(self) -> dict[str, JsonValue]:
        """JSON-compatible representation, including ``duration_seconds``."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> TestResult:
        """Rebuild a result from :meth:`to_dict` output; ``duration_seconds`` is recomputed."""
        payload = dict(data)
        payload.pop("duration_seconds", None)
        return cls.model_validate(payload)
