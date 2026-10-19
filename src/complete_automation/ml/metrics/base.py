"""Metric registry.

A metric is a function ``fn(context) -> float | MetricValue | list[MetricValue]`` registered with
``@metric(task, name, direction)``. It signals "cannot be computed" by raising
:class:`MetricUndefined` (the data make it undefined) or :class:`MetricUnsupported` (the task or
available model outputs cannot provide it). Both become explicit :class:`MetricValue` results
with a reason; no placeholder number is ever substituted.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from complete_automation.ml.errors import MLConfigurationError
from complete_automation.ml.schemas import (
    Direction,
    EvaluationConfig,
    Label,
    MetricStatus,
    MetricValue,
    TaskType,
)


class MetricUndefined(Exception):  # noqa: N818
    """The metric is mathematically undefined for this data."""


class MetricUnsupported(Exception):  # noqa: N818
    """The metric needs an input the task or the model does not provide."""


@dataclass(frozen=True)
class MetricContext:
    config: EvaluationConfig
    y_true: np.ndarray | None
    y_pred: np.ndarray
    classes: list[Label]
    probabilities: np.ndarray | None = None
    """Columns in ``classes`` order."""
    scores: np.ndarray | None = None
    features: np.ndarray | None = None
    """Numeric feature matrix, for intrinsic clustering metrics."""

    def need_truth(self, metric: str) -> np.ndarray:
        if self.y_true is None:
            raise MetricUnsupported(f"{metric} needs reference labels or targets")
        return self.y_true


MetricFn = Callable[[MetricContext], "float | MetricValue | list[MetricValue]"]


@dataclass(frozen=True)
class MetricSpec:
    task: TaskType
    name: str
    direction: Direction
    fn: MetricFn
    description: str


_REGISTRY: dict[tuple[TaskType, str], MetricSpec] = {}


def metric(
    task: TaskType, name: str, direction: Direction, description: str = ""
) -> Callable[[MetricFn], MetricFn]:
    """Register a metric. Re-registering a name is an error; pick a new name."""

    def register(fn: MetricFn) -> MetricFn:
        key = (task, name)
        if key in _REGISTRY:
            raise MLConfigurationError(f"metric {name!r} is already registered for {task.value}")
        _REGISTRY[key] = MetricSpec(task, name, direction, fn, description or (fn.__doc__ or "").strip())
        return fn

    return register


def available_metrics(task: TaskType) -> dict[str, MetricSpec]:
    return {name: spec for (t, name), spec in _REGISTRY.items() if t is task}


def get_metric(task: TaskType, name: str) -> MetricSpec:
    try:
        return _REGISTRY[(task, name)]
    except KeyError:
        known = sorted(available_metrics(task))
        raise MLConfigurationError(
            f"metric {name!r} is not available for {task.value} tasks; available: {known}"
        ) from None


def direction_of(task: TaskType, metric_name: str) -> Direction:
    """Comparison direction of a metric; ``recall[fraud]`` uses the direction of ``recall``."""
    base = metric_name.split("[", 1)[0]
    spec = _REGISTRY.get((task, base))
    return spec.direction if spec else Direction.NONE


def run_metric(spec: MetricSpec, context: MetricContext) -> list[MetricValue]:
    """Execute one metric; failures to compute become undefined/unsupported results."""
    try:
        with quiet():
            raw = spec.fn(context)
    except MetricUndefined as reason:
        return [_missing(spec, MetricStatus.UNDEFINED, str(reason))]
    except MetricUnsupported as reason:
        return [_missing(spec, MetricStatus.UNSUPPORTED, str(reason))]
    except ValueError as error:  # a library precondition we did not anticipate: undefined, with its text
        return [_missing(spec, MetricStatus.UNDEFINED, f"{type(error).__name__}: {error}")]
    items = raw if isinstance(raw, list) else [raw]
    out: list[MetricValue] = []
    for item in items:
        if isinstance(item, MetricValue):
            out.append(item)
            continue
        value = float(item)
        if not math.isfinite(value):
            out.append(_missing(spec, MetricStatus.UNDEFINED, f"computed value is {value}"))
        else:
            out.append(MetricValue(name=spec.name, status=MetricStatus.COMPUTED, value=value, direction=spec.direction))
    return out


def _missing(spec: MetricSpec, status: MetricStatus, reason: str) -> MetricValue:
    return MetricValue(name=spec.name, status=status, direction=spec.direction, reason=reason)


@contextmanager
def quiet() -> Any:
    """Silence library warnings: every condition they describe is reported explicitly instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
