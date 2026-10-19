"""Metric registry and the built-in classification, regression and clustering metrics."""

from complete_automation.ml.metrics import classification, clustering, regression
from complete_automation.ml.metrics.base import (
    MetricContext,
    MetricSpec,
    MetricUndefined,
    MetricUnsupported,
    available_metrics,
    direction_of,
    get_metric,
    metric,
    run_metric,
)

__all__ = [
    "MetricContext",
    "MetricSpec",
    "MetricUndefined",
    "MetricUnsupported",
    "available_metrics",
    "classification",
    "clustering",
    "direction_of",
    "get_metric",
    "metric",
    "regression",
    "run_metric",
]
