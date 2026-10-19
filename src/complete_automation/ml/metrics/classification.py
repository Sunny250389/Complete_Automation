"""Classification metrics.

Behaviour worth knowing, especially for imbalanced data:

* ``accuracy`` rewards predicting the majority class; always read it next to ``balanced_accuracy``
  (mean recall over the classes present in the targets) and the per-class metrics.
* ``precision``/``recall``/``f1`` use ``config.averaging`` (default ``macro``: every class counts
  equally, so a rare class matters as much as a common one). ``precision_macro``, ``f1_weighted``,
  ``recall_binary`` and so on select an averaging explicitly. ``binary`` needs
  ``config.positive_label``; the framework never guesses the positive class.
* A 0/0 precision, recall or F1 follows ``config.zero_division``: counted as 0 (and listed in a
  warning by the evaluator) or reported as undefined.
* ``roc_auc``, ``average_precision`` and ``log_loss`` need probabilities or decision scores and are
  *unsupported* when only hard labels exist. Multiclass ``roc_auc`` and ``average_precision`` are
  one-vs-rest macro averages and are undefined if any class is absent from the targets.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    log_loss as sk_log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from complete_automation.ml.metrics.base import (
    MetricContext,
    MetricUndefined,
    MetricUnsupported,
    metric,
)
from complete_automation.ml.schemas import Direction, MetricStatus, MetricValue, TaskType

T = TaskType.CLASSIFICATION
HIGHER, NONE = Direction.HIGHER_IS_BETTER, Direction.NONE
AVERAGES = ("binary", "macro", "micro", "weighted")
_PRF = {"precision": 0, "recall": 1, "f1": 2}


def _zero_division(ctx: MetricContext) -> float:
    return np.nan if ctx.config.zero_division == "undefined" else 0.0


def _prf_per_class(ctx: MetricContext, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = ctx.need_truth(name)
    return precision_recall_fscore_support(
        y_true, ctx.y_pred, labels=ctx.classes, average=None, zero_division=_zero_division(ctx)
    )


def _scalar(value: float) -> float:
    if value is None or np.isnan(value):
        raise MetricUndefined("0/0: no samples to compute the ratio from")
    return float(value)


def _prf(ctx: MetricContext, which: str, average: str) -> float:
    y_true = ctx.need_truth(which)
    index = _PRF[which]
    if average == "micro":
        out = precision_recall_fscore_support(
            y_true, ctx.y_pred, labels=ctx.classes, average="micro", zero_division=_zero_division(ctx)
        )
        return _scalar(out[index])
    values = _prf_per_class(ctx, which)[index]
    support = _prf_per_class(ctx, which)[3]
    if average == "binary":
        positive = ctx.config.positive_label
        if len(ctx.classes) != 2:
            raise MetricUnsupported(f"binary averaging needs exactly two classes, found {len(ctx.classes)}")
        if positive is None or positive not in ctx.classes:
            raise MetricUnsupported("binary averaging needs config.positive_label to be one of the classes")
        return _scalar(values[ctx.classes.index(positive)])
    if average == "macro":
        undefined = [c for c, v in zip(ctx.classes, values) if np.isnan(v)]
        if undefined:
            raise MetricUndefined(f"{which} is undefined (0/0) for class(es) {undefined}")
        return float(values.mean())
    present = support > 0
    if not present.any():
        raise MetricUndefined("no samples")
    if np.isnan(values[present]).any():
        raise MetricUndefined(f"{which} is undefined (0/0) for a class with samples")
    return float((values[present] * support[present]).sum() / support[present].sum())


def _register_prf() -> None:
    for which in _PRF:
        metric(T, which, HIGHER, f"{which} using config.averaging")(
            lambda ctx, which=which: _prf(ctx, which, ctx.config.averaging)
        )
        for average in AVERAGES:
            metric(T, f"{which}_{average}", HIGHER, f"{average}-averaged {which}")(
                lambda ctx, which=which, average=average: _prf(ctx, which, average)
            )


_register_prf()


@metric(T, "accuracy", HIGHER, "share of exact matches")
def accuracy(ctx: MetricContext) -> float:
    y_true = ctx.need_truth("accuracy")
    return float(np.mean(np.asarray(y_true == ctx.y_pred, dtype=bool)))


@metric(T, "balanced_accuracy", HIGHER, "mean recall over the classes present in the targets")
def balanced_accuracy(ctx: MetricContext) -> float:
    _, recall, _, support = _prf_per_class(ctx, "balanced_accuracy")
    present = support > 0
    if not present.any():
        raise MetricUndefined("no samples")
    return float(np.nan_to_num(recall[present]).mean())


@metric(T, "confusion_matrix", NONE, "rows are true classes, columns predicted classes")
def confusion_matrix(ctx: MetricContext) -> MetricValue:
    matrix = sk_confusion_matrix(ctx.need_truth("confusion_matrix"), ctx.y_pred, labels=ctx.classes)
    return MetricValue(
        name="confusion_matrix",
        status=MetricStatus.COMPUTED,
        detail={"labels": list(ctx.classes), "matrix": matrix.tolist()},
    )


@metric(T, "per_class", NONE, "precision, recall, F1 and support for every class")
def per_class(ctx: MetricContext) -> list[MetricValue]:
    precision, recall, f1, support = _prf_per_class(ctx, "per_class")
    out: list[MetricValue] = []
    for i, label in enumerate(ctx.classes):
        for name, array, direction in (
            ("precision", precision, HIGHER),
            ("recall", recall, HIGHER),
            ("f1", f1, HIGHER),
        ):
            key = f"{name}[{label}]"
            if np.isnan(array[i]):
                out.append(MetricValue(name=key, status=MetricStatus.UNDEFINED, direction=direction,
                                       reason=f"{name} is 0/0 for class {label!r}"))
            else:
                out.append(MetricValue(name=key, status=MetricStatus.COMPUTED, value=float(array[i]), direction=direction))
        out.append(MetricValue(name=f"support[{label}]", status=MetricStatus.COMPUTED,
                               value=float(support[i]), direction=NONE))
    return out


def zero_division_notes(ctx: MetricContext) -> list[str]:
    """Classes whose precision or recall is 0/0, for the warning the evaluator raises under ``zero_division='zero'``."""
    if ctx.y_true is None:
        return []
    notes = []
    for label in ctx.classes:
        if not np.any(ctx.y_pred == label):
            notes.append(f"class {label!r} is never predicted (precision is 0/0)")
        if not np.any(ctx.y_true == label):
            notes.append(f"class {label!r} has no samples in the targets (recall is 0/0)")
    return notes


# --- score based metrics ----------------------------------------------------------------------


def _score_matrix(ctx: MetricContext, name: str) -> np.ndarray:
    if ctx.probabilities is not None:
        return ctx.probabilities
    if ctx.scores is not None:
        # a 1-D score of a binary problem favours the second class, like scikit-learn's decision_function
        return np.column_stack([-ctx.scores, ctx.scores]) if ctx.scores.ndim == 1 else ctx.scores
    raise MetricUnsupported(f"{name} needs predict_proba or decision_function; hard class labels are not enough")


def _one_vs_rest(ctx: MetricContext, name: str, score_fn) -> float:  # type: ignore[no-untyped-def]
    y_true = ctx.need_truth(name)
    scores = _score_matrix(ctx, name)
    if len(ctx.classes) < 2:
        raise MetricUndefined("fewer than two classes")
    if len(ctx.classes) == 2:
        positive = ctx.config.positive_label
        if positive is None or positive not in ctx.classes:
            raise MetricUnsupported(f"binary {name} needs config.positive_label to be one of the classes")
        indicator = np.asarray(y_true == positive, dtype=bool)
        if indicator.all() or not indicator.any():
            raise MetricUndefined("only one class is present in the targets")
        return float(score_fn(indicator, scores[:, ctx.classes.index(positive)]))
    values, absent = [], []
    for i, label in enumerate(ctx.classes):
        indicator = np.asarray(y_true == label, dtype=bool)
        if indicator.all() or not indicator.any():
            absent.append(label)
            continue
        values.append(score_fn(indicator, scores[:, i]))
    if absent:
        raise MetricUndefined(f"class(es) {absent} are absent from (or are the only class in) the targets")
    return float(np.mean(values))


@metric(T, "roc_auc", HIGHER, "area under the ROC curve (one-vs-rest macro for multiclass)")
def roc_auc(ctx: MetricContext) -> float:
    return _one_vs_rest(ctx, "roc_auc", roc_auc_score)


@metric(T, "average_precision", HIGHER, "area under the precision-recall curve (one-vs-rest macro for multiclass)")
def average_precision(ctx: MetricContext) -> float:
    return _one_vs_rest(ctx, "average_precision", average_precision_score)


@metric(T, "log_loss", Direction.LOWER_IS_BETTER, "cross-entropy of the predicted probabilities")
def log_loss(ctx: MetricContext) -> float:
    y_true = ctx.need_truth("log_loss")
    if ctx.probabilities is None:
        raise MetricUnsupported("log_loss needs predict_proba; hard labels and decision scores are not probabilities")
    return float(sk_log_loss(y_true, ctx.probabilities, labels=ctx.classes))


DEFAULT_LABEL_METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "confusion_matrix", "per_class",
)
DEFAULT_SCORE_METRICS = ("roc_auc", "average_precision")
DEFAULT_PROBABILITY_METRICS = ("log_loss",)
