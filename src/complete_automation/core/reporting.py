"""Reporter interface, run summary and the JSON reporter.

A reporter collects :class:`TestResult` records and presents them when the run
is finalized. Reporters never recompute domain metrics; they count statuses
and pass metrics and artifact references through unchanged.

Lifecycle: ``start_run(run)`` once, ``publish(result)`` per result, then
``finalize(run)`` once. Publishing after finalization, or failing to write the
report, raises :class:`ReportingError`.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from complete_automation.core.context import RunContext
from complete_automation.core.exceptions import ReportingError
from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor
from complete_automation.core.results import SCHEMA_VERSION, Status, TestResult
from complete_automation.core.serialization import dumps_deterministic


class StatusCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error: int = 0
    incomplete: int = 0


class RunSummary(StatusCounts):
    duration_seconds: float = 0.0
    """Sum of test durations (not wall-clock time when tests run in parallel)."""
    artifact_count: int = 0
    metric_count: int = 0
    by_module: dict[str, StatusCounts] = Field(default_factory=dict)


def _discard(path: Path) -> None:
    """Best-effort removal of a leftover temporary file.

    Never raises: the write failure that led here is the error worth reporting,
    and which ``OSError`` a missing parent produces differs by platform
    (``FileNotFoundError`` on Windows, ``NotADirectoryError`` on Linux).
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _counts(results: list[TestResult]) -> dict[str, int]:
    tally = Counter(result.status.value for result in results)
    return {"total": len(results), **{status.value: tally[status.value] for status in Status}}


def summarize(results: Iterable[TestResult]) -> RunSummary:
    items = list(results)
    modules: dict[str, list[TestResult]] = {}
    for result in items:
        modules.setdefault(result.module, []).append(result)
    return RunSummary(
        **_counts(items),
        duration_seconds=round(sum(r.duration_seconds for r in items), 6),
        artifact_count=sum(len(r.artifacts) for r in items),
        metric_count=sum(len(r.metrics) for r in items),
        by_module={name: StatusCounts(**_counts(group)) for name, group in sorted(modules.items())},
    )


@runtime_checkable
class Reporter(Protocol):
    def start_run(self, run: RunContext) -> None: ...

    def publish(self, result: TestResult) -> None: ...

    def finalize(self, run: RunContext | None = None) -> Path | None:
        """Write the report; return its location, or ``None`` for non-file reporters."""
        ...


class JsonReporter:
    """Writes one JSON document per run: ``{schema_version, run, summary, results}``.

    The file is written atomically (temporary file, then rename), with sorted
    keys, and passed through redaction as a last line of defence.
    """

    def __init__(self, output_path: Path, *, redactor: Redactor | None = None) -> None:
        self.output_path = Path(output_path)
        self.redactor = redactor or DEFAULT_REDACTOR
        self._run: RunContext | None = None
        self._results: list[TestResult] = []
        self._finalized = False
        self._lock = threading.Lock()

    @property
    def results(self) -> tuple[TestResult, ...]:
        with self._lock:
            return tuple(self._results)

    def start_run(self, run: RunContext) -> None:
        with self._lock:
            self._ensure_open()
            self._run = run

    def publish(self, result: TestResult) -> None:
        if not isinstance(result, TestResult):
            raise ReportingError(f"expected TestResult, got {type(result).__name__}")
        with self._lock:
            self._ensure_open()
            self._results.append(result)

    def finalize(self, run: RunContext | None = None) -> Path:
        with self._lock:
            self._ensure_open()
            self._finalized = True
            run = run or self._run
            document = {
                "schema_version": SCHEMA_VERSION,
                "run": run.model_dump(mode="json") if run else None,
                "summary": summarize(self._results).model_dump(mode="json"),
                "results": [result.to_dict() for result in self._results],
            }
        try:
            text = dumps_deterministic(self.redactor.redact(document))
        except (TypeError, ValueError) as error:
            raise ReportingError(f"report is not JSON-serializable: {error}") from error
        temporary = self.output_path.with_name(f".{self.output_path.name}.tmp")
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.output_path)
        except OSError as error:
            _discard(temporary)
            raise ReportingError(f"cannot write JSON report to {self.output_path}") from error
        return self.output_path

    def _ensure_open(self) -> None:
        if self._finalized:
            raise ReportingError("reporter is already finalized")
