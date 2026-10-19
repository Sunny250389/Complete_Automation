"""Baseline comparison: stored metrics of an earlier evaluation versus the current ones.

A baseline is a small JSON file with metric values only (no data, no predictions). Comparison
always reports absolute and relative differences; a change alone is never a failure. Only a
:class:`BaselineRule` (an explicit allowance for getting *worse*, in the metric's own direction)
can fail it. Undefined values on either side make a ruled metric ``unavailable``, which fails a
required rule: the comparison cannot confirm there was no regression.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore
from complete_automation.core.datasets import DataLoader
from complete_automation.core.exceptions import TestDataError
from complete_automation.ml.errors import BaselineError
from complete_automation.ml.metrics import direction_of
from complete_automation.ml.schemas import (
    BaselineComparison,
    BaselineRule,
    Direction,
    EvaluationReport,
    MetricComparison,
    MetricValue,
    TaskType,
)


class Baseline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    baseline_id: str = Field(min_length=1)
    task: TaskType
    model_id: str
    dataset_id: str
    dataset_fingerprint: str
    metrics: dict[str, float | None]
    """Scalar metric values; ``None`` records a metric that was undefined."""

    @classmethod
    def from_report(cls, report: EvaluationReport, baseline_id: str | None = None) -> Baseline:
        return cls(
            baseline_id=baseline_id or f"{report.model.model_id}@{report.dataset.dataset_id}",
            task=report.task,
            model_id=report.model.model_id,
            dataset_id=report.dataset.dataset_id,
            dataset_fingerprint=report.dataset.fingerprint,
            metrics={m.name: m.value for m in report.metrics if m.is_scalar or m.value is None and m.detail is None},
        )

    def save(self, store: ArtifactStore, name: str | None = None) -> ArtifactRef:
        """Store the baseline as an artifact; copy it into the baseline directory to adopt it."""
        return store.write_json(
            name or f"baseline-{self.baseline_id}.json", self.model_dump(mode="json"), subdir="ml"
        )

    @classmethod
    def load(cls, root: Path, relative: str | Path) -> Baseline:
        """Read a baseline file located under ``root``."""
        try:
            data = DataLoader(root).load(relative)
        except TestDataError as error:
            raise BaselineError(str(error)) from error
        try:
            return cls.model_validate(data)
        except ValueError as error:
            raise BaselineError(f"invalid baseline file {relative}: {error}") from error

    @classmethod
    def loads(cls, text: str) -> Baseline:
        try:
            return cls.model_validate(json.loads(text))
        except ValueError as error:
            raise BaselineError(f"invalid baseline: {error}") from error


def compare_to_baseline(
    baseline: Baseline,
    current: Iterable[MetricValue],
    rules: Iterable[BaselineRule] = (),
    *,
    task: TaskType,
) -> BaselineComparison:
    if baseline.task is not task:
        raise BaselineError(f"baseline is for {baseline.task.value} but the evaluation is {task.value}")
    now = {m.name: m.value for m in current if m.is_scalar or m.value is None and m.detail is None}
    ruled = {rule.metric: rule for rule in rules}
    for name in ruled:
        if name not in baseline.metrics and name not in now:
            raise BaselineError(f"baseline rule for {name!r}: metric is in neither baseline nor evaluation")
    comparisons = [
        _compare(name, baseline.metrics.get(name), now.get(name), ruled.get(name), task)
        for name in sorted(set(baseline.metrics) | set(now))
    ]
    failed = any(
        c.outcome in ("regressed", "unavailable") and ruled[c.metric].required
        for c in comparisons
        if c.metric in ruled
    )
    return BaselineComparison(baseline_id=baseline.baseline_id, comparisons=tuple(comparisons), passed=not failed)


def _compare(
    name: str, before: float | None, after: float | None, rule: BaselineRule | None, task: TaskType
) -> MetricComparison:
    difference = None if before is None or after is None else after - before
    relative = None if difference is None or before == 0 else difference / abs(before)
    if rule is None:
        outcome = "unchanged" if difference == 0 else "changed"
        return MetricComparison(metric=name, baseline=before, current=after, absolute_difference=difference,
                                relative_difference=relative, outcome=outcome)
    direction = direction_of(task, name)
    if direction is Direction.NONE:
        raise BaselineError(f"baseline rule for {name!r}: the metric has no better/worse direction")
    if difference is None:
        side = "baseline" if before is None else "current"
        return MetricComparison(metric=name, baseline=before, current=after, absolute_difference=None,
                                relative_difference=None, outcome="unavailable",
                                reason=f"the {side} value is missing or undefined")
    worse_by = -difference if direction is Direction.HIGHER_IS_BETTER else difference
    reasons = []
    if rule.max_drop is not None and worse_by > rule.max_drop + 1e-12:
        reasons.append(f"worsened by {worse_by:.10g}, allowed {rule.max_drop:.10g}")
    if rule.max_relative_drop is not None:
        if relative is None:
            return MetricComparison(metric=name, baseline=before, current=after, absolute_difference=difference,
                                    relative_difference=None, outcome="unavailable",
                                    reason="relative change is undefined because the baseline is 0")
        if worse_by / abs(before) > rule.max_relative_drop + 1e-12:  # type: ignore[operator]
            reasons.append(f"worsened by {worse_by / abs(before):.2%}, allowed {rule.max_relative_drop:.2%}")  # type: ignore[operator]
    return MetricComparison(
        metric=name, baseline=before, current=after, absolute_difference=difference,
        relative_difference=relative, outcome="regressed" if reasons else "passed",
        reason="; ".join(reasons) or None,
    )
