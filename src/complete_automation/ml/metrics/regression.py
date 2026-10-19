"""Regression metrics.

Limitations:

* ``r2`` compares the model with predicting the mean target. It is undefined when the targets are
  constant, can be negative for a model worse than the mean, and says nothing about error size.
  It is not comparable across datasets.
* ``mape`` is a *fraction* (0.10 means 10 %). It explodes for targets near zero and is undefined
  when any target is zero (``|y| <= config.mape_epsilon``) unless ``mape_zero_policy='exclude'``,
  which drops those rows from this metric only and records how many.
* ``mse``/``rmse`` weight large errors heavily; ``mae`` and ``median_ae`` are more robust to outliers.
* ``rmse``, ``mae`` and ``mse`` are in the target's units.
"""

from __future__ import annotations

import numpy as np

from complete_automation.ml.metrics.base import MetricContext, MetricUndefined, metric
from complete_automation.ml.schemas import Direction, MetricStatus, MetricValue, TaskType

T = TaskType.REGRESSION
LOWER, HIGHER = Direction.LOWER_IS_BETTER, Direction.HIGHER_IS_BETTER


def _arrays(ctx: MetricContext, name: str) -> tuple[np.ndarray, np.ndarray]:
    return ctx.need_truth(name).astype(float), ctx.y_pred.astype(float)


@metric(T, "mae", LOWER, "mean absolute error")
def mae(ctx: MetricContext) -> float:
    y, p = _arrays(ctx, "mae")
    return float(np.mean(np.abs(y - p)))


@metric(T, "mse", LOWER, "mean squared error")
def mse(ctx: MetricContext) -> float:
    y, p = _arrays(ctx, "mse")
    return float(np.mean((y - p) ** 2))


@metric(T, "rmse", LOWER, "root mean squared error")
def rmse(ctx: MetricContext) -> float:
    y, p = _arrays(ctx, "rmse")
    return float(np.sqrt(np.mean((y - p) ** 2)))


@metric(T, "median_ae", LOWER, "median absolute error")
def median_ae(ctx: MetricContext) -> float:
    y, p = _arrays(ctx, "median_ae")
    return float(np.median(np.abs(y - p)))


@metric(T, "r2", HIGHER, "coefficient of determination")
def r2(ctx: MetricContext) -> float:
    y, p = _arrays(ctx, "r2")
    if len(y) < 2:
        raise MetricUndefined("r2 needs at least two samples")
    total = float(np.sum((y - y.mean()) ** 2))
    if total == 0.0:
        raise MetricUndefined("r2 is undefined: the targets are constant")
    return 1.0 - float(np.sum((y - p) ** 2)) / total


@metric(T, "mape", LOWER, "mean absolute percentage error, as a fraction")
def mape(ctx: MetricContext) -> float | MetricValue:
    y, p = _arrays(ctx, "mape")
    zero = np.abs(y) <= ctx.config.mape_epsilon
    count = int(zero.sum())
    if count and ctx.config.mape_zero_policy == "undefined":
        raise MetricUndefined(
            f"{count} of {len(y)} target(s) are zero or near zero; "
            "set mape_zero_policy='exclude' to leave them out of this metric"
        )
    keep = ~zero
    if not keep.any():
        raise MetricUndefined("every target is zero or near zero")
    value = float(np.mean(np.abs((y[keep] - p[keep]) / y[keep])))
    if count:
        return MetricValue(
            name="mape", status=MetricStatus.COMPUTED, value=value, direction=LOWER,
            detail={"excluded_zero_targets": count, "samples_used": int(keep.sum())},
        )
    return value


DEFAULT_METRICS = ("mae", "mse", "rmse", "r2", "mape", "median_ae")
