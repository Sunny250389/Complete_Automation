"""Dataset and prediction validation.

Findings are *blocking* (``error``) or advisory (``warning``/``info``). They carry counts and
column names, never cell values. Invalid rows are never dropped silently: with
``invalid_rows='exclude'`` they are removed and a ``rows_excluded`` finding says how many and why.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from complete_automation.ml.data import Dataset, column_is, is_numeric_column, to_python
from complete_automation.ml.models import Predictions
from complete_automation.ml.schemas import (
    EvaluationConfig,
    Label,
    Severity,
    TaskType,
    ValidationFinding,
)

_E, _W, _I = Severity.ERROR, Severity.WARNING, Severity.INFO


def _finding(code: str, severity: Severity, message: str, **details: Any) -> ValidationFinding:
    return ValidationFinding(code=code, severity=severity, message=message, details=details)


def resolve_classes(
    y_true: np.ndarray | None, y_pred: np.ndarray | None, config: EvaluationConfig
) -> list[Label]:
    """Declared labels, else every label seen in the targets and predictions (sorted)."""
    if config.labels is not None:
        return list(config.labels)
    seen: set[Any] = set()
    for array in (y_true, y_pred):
        if array is not None:
            seen.update(to_python(v) for v in pd.unique(array) if not pd.isna(v))
    try:
        return sorted(seen)
    except TypeError:
        return sorted(seen, key=lambda v: (type(v).__name__, str(v)))


# --- dataset ----------------------------------------------------------------------------------


def validate_dataset(dataset: Dataset, config: EvaluationConfig) -> list[ValidationFinding]:
    frame, schema = dataset.frame, dataset.schema
    findings: list[ValidationFinding] = []
    if len(frame) == 0:
        return [_finding("empty_dataset", _E, "the dataset has no rows")]

    needs_target = config.task is not TaskType.CLUSTERING
    if needs_target and dataset.target is None:
        return [_finding("missing_target", _E, f"{config.task.value} evaluation needs a target column")]
    required = list(dataset.features)
    if dataset.target is not None:
        required.append(dataset.target)
    if schema is not None:
        required += list(schema.columns) + list(schema.unique_key or ())
    required = list(dict.fromkeys(required))
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return [_finding("missing_columns", _E, f"required columns are missing: {missing}", columns=missing)]

    # In exclude mode these row-level problems are warnings; the exclusion step reports the counts.
    row_severity = _W if config.invalid_rows == "exclude" else _E

    if schema is not None:
        for name, spec in schema.columns.items():
            if not column_is(frame[name], spec.dtype):
                findings.append(_finding(
                    "unexpected_dtype", _E,
                    f"column {name!r} is not {spec.dtype} (found {frame[name].dtype})",
                    column=name, expected=spec.dtype, found=str(frame[name].dtype)))
                continue
            series = frame[name].dropna()
            if spec.allowed_values is not None:
                bad = int((~series.isin(spec.allowed_values)).sum())
                if bad:
                    findings.append(_finding(
                        "unexpected_values", _E,
                        f"column {name!r} has {bad} value(s) outside its allowed set",
                        column=name, count=bad))
            if (spec.min is not None or spec.max is not None) and is_numeric_column(series):
                bad = int(((series < spec.min) if spec.min is not None else False).sum()
                          + ((series > spec.max) if spec.max is not None else False).sum())
                if bad:
                    findings.append(_finding(
                        "out_of_range", _E, f"column {name!r} has {bad} value(s) out of range",
                        column=name, count=bad))
        if schema.forbid_extra_columns:
            known = set(schema.columns) | set(dataset.features) | {dataset.target}
            extra = [c for c in frame.columns if c not in known]
            if extra:
                findings.append(_finding("unexpected_columns", _E, f"unexpected columns: {extra}", columns=extra))

    for name in required:
        series = frame[name]
        spec = schema.columns.get(name) if schema else None
        nulls = int(series.isna().sum())
        if nulls:
            nullable = spec.nullable if spec else name != dataset.target
            if not nullable:
                findings.append(_finding(
                    "null_values", row_severity, f"column {name!r} has {nulls} missing value(s)",
                    column=name, count=nulls))
            elif spec is None:
                findings.append(_finding(
                    "null_values", _W, f"feature {name!r} has {nulls} missing value(s)",
                    column=name, count=nulls))
        if is_numeric_column(series):
            infinite = int(np.isinf(series.to_numpy(dtype=float)).sum())
            if infinite:
                findings.append(_finding(
                    "non_finite_values", row_severity,
                    f"column {name!r} has {infinite} infinite value(s)", column=name, count=infinite))

    if config.task is TaskType.REGRESSION and not is_numeric_column(frame[dataset.target]):
        findings.append(_finding("non_numeric_target", _E, f"regression target {dataset.target!r} is not numeric"))
    if config.task is TaskType.CLUSTERING:
        text = [c for c in dataset.features if not is_numeric_column(frame[c])]
        if text:
            findings.append(_finding("non_numeric_features", _E, f"clustering needs numeric features: {text}", columns=text))

    mode = schema.duplicates if schema else "warn"
    if mode != "off":
        key = list(schema.unique_key) if schema and schema.unique_key else None
        duplicated = int(frame.duplicated(subset=key).sum())
        if duplicated:
            findings.append(_finding(
                "duplicate_records", _E if mode == "error" else _W,
                f"{duplicated} duplicate record(s)", count=duplicated, key=key or "all columns"))

    if config.task is TaskType.CLASSIFICATION:
        findings += _class_findings(frame[dataset.target].dropna(), config)
    return findings


def _class_findings(target: pd.Series, config: EvaluationConfig) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    if config.labels is not None:
        bad = int((~target.isin(config.labels)).sum())
        if bad:
            findings.append(_finding(
                "invalid_target_labels", _E, f"{bad} target value(s) are not in the declared labels", count=bad))
    counts = target.value_counts()
    if counts.empty:
        return findings
    distribution = {str(k): int(v) for k, v in counts.items()}
    findings.append(_finding("class_distribution", _I, "class distribution of the target", counts=distribution))
    if len(counts) == 1:
        findings.append(_finding("single_class_target", _W, "the target has a single class; many metrics are undefined"))
    else:
        ratio = float(counts.max() / counts.min())
        if ratio >= config.imbalance_warning_ratio:
            findings.append(_finding(
                "class_imbalance", _W,
                f"majority/minority class ratio is {ratio:.1f}; accuracy can be misleading",
                ratio=round(ratio, 3), counts=distribution))
    return findings


def invalid_row_mask(dataset: Dataset) -> tuple[np.ndarray, dict[str, int]]:
    """Rows with a missing target, a missing feature or a non-finite number, and the count per reason."""
    frame = dataset.frame
    reasons: dict[str, np.ndarray] = {}
    if dataset.target is not None:
        reasons["missing_target"] = frame[dataset.target].isna().to_numpy()
    columns = list(dataset.features) + ([dataset.target] if dataset.target else [])
    reasons["missing_feature"] = frame[list(dataset.features)].isna().any(axis=1).to_numpy()
    numeric = [c for c in columns if is_numeric_column(frame[c])]
    inf = np.zeros(len(frame), dtype=bool)
    for column in numeric:
        inf |= np.isinf(frame[column].to_numpy(dtype=float))
    reasons["non_finite_value"] = inf
    mask = np.zeros(len(frame), dtype=bool)
    for flags in reasons.values():
        mask |= flags
    return mask, {k: int(v.sum()) for k, v in reasons.items() if v.any()}


# --- predictions ------------------------------------------------------------------------------


def validate_predictions(
    predictions: Predictions, n_rows: int, classes: list[Label], config: EvaluationConfig
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    labels = predictions.labels
    if labels.ndim != 1:
        return [_finding(
            "unexpected_prediction_shape", _E,
            f"predictions have shape {list(labels.shape)}; expected ({n_rows},)",
            shape=list(labels.shape), expected=[n_rows])]
    if len(labels) != n_rows:
        return [_finding(
            "prediction_length_mismatch", _E,
            f"{len(labels)} prediction(s) for {n_rows} row(s)", predictions=len(labels), rows=n_rows)]
    missing = int(pd.isna(labels).sum())
    if missing:
        findings.append(_finding("missing_predictions", _E, f"{missing} prediction(s) are missing", count=missing))
    present = labels[~pd.isna(labels)] if missing else labels

    if config.task is TaskType.REGRESSION:
        if not np.issubdtype(present.dtype, np.number) or present.dtype == bool:
            findings.append(_finding("non_numeric_predictions", _E, "regression predictions are not numeric"))
        elif (bad := int((~np.isfinite(present.astype(float))).sum())):
            findings.append(_finding("non_finite_predictions", _E, f"{bad} prediction(s) are not finite", count=bad))
    elif config.task is TaskType.CLUSTERING:
        numeric = np.issubdtype(present.dtype, np.number) and present.dtype != bool
        if not numeric or not np.all(np.isfinite(present.astype(float))) or np.any(present.astype(float) % 1):
            findings.append(_finding("invalid_cluster_assignments", _E, "cluster assignments must be integers"))
    else:
        if config.labels is not None:
            bad = int((~pd.Series(present).isin(config.labels)).sum())
            if bad:
                findings.append(_finding(
                    "unexpected_predicted_labels", _E,
                    f"{bad} prediction(s) are not in the declared labels", count=bad))
        findings += _probability_findings(predictions, n_rows, classes, config)
    return findings


def _probability_findings(
    predictions: Predictions, n_rows: int, classes: list[Label], config: EvaluationConfig
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    proba, scores = predictions.probabilities, predictions.scores
    if proba is not None:
        expected = [n_rows, len(classes)]
        if proba.ndim != 2 or list(proba.shape) != expected:
            findings.append(_finding(
                "invalid_probability_shape", _E,
                f"probabilities have shape {list(proba.shape)}; expected {expected} (rows, classes)",
                shape=list(proba.shape), expected=expected))
        elif predictions.classes is not None and set(predictions.classes) != set(classes):
            findings.append(_finding(
                "probability_classes_mismatch", _E,
                "the model's classes differ from the evaluated classes",
                model_classes=[str(c) for c in predictions.classes], classes=[str(c) for c in classes]))
        elif not np.isfinite(proba).all():
            findings.append(_finding("non_finite_probabilities", _E, "probabilities contain NaN or infinity"))
        else:
            out = int(((proba < 0) | (proba > 1)).any(axis=1).sum())
            if out:
                findings.append(_finding("probability_out_of_range", _E, f"{out} row(s) have probabilities outside [0, 1]", rows=out))
            off = int((~np.isclose(proba.sum(axis=1), 1.0, atol=config.probability_atol, rtol=0)).sum())
            if off:
                findings.append(_finding("probabilities_not_normalized", _E, f"{off} row(s) do not sum to 1", rows=off))
    if scores is not None:
        valid_shapes = [[n_rows]] if len(classes) == 2 else []
        valid_shapes.append([n_rows, len(classes)])
        if list(scores.shape) not in valid_shapes:
            findings.append(_finding(
                "invalid_score_shape", _E,
                f"decision scores have shape {list(scores.shape)}; expected one of {valid_shapes}",
                shape=list(scores.shape)))
        elif not np.isfinite(scores).all():
            findings.append(_finding("non_finite_scores", _E, "decision scores contain NaN or infinity"))
    return findings


def align_columns(array: np.ndarray, model_classes: tuple[Label, ...] | None, classes: list[Label]) -> np.ndarray:
    """Reorder probability/score columns from the model's class order to ``classes`` order."""
    if array.ndim != 2 or model_classes is None or list(model_classes) == classes:
        return array
    order = [list(model_classes).index(c) for c in classes]
    return array[:, order]
