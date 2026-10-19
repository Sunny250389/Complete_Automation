"""Builders shared by the ML unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from complete_automation.ml import (
    Dataset,
    EvaluationConfig,
    EvaluationReport,
    Evaluator,
    PrecomputedAdapter,
)


def evaluate_arrays(
    task: str,
    y_true,
    y_pred,
    *,
    proba=None,
    scores=None,
    classes=None,
    features=None,
    **config,
) -> EvaluationReport:
    """Evaluate precomputed predictions for ``y_true`` (``None`` for clustering without reference)."""
    n = len(y_pred)
    frame = pd.DataFrame({"x": np.arange(n, dtype=float)} if features is None else features)
    cols = list(frame.columns)
    if y_true is not None:
        frame["target"] = list(y_true)
    dataset = Dataset.from_frame(frame, "unit", features=cols, target="target" if y_true is not None else None)
    adapter = PrecomputedAdapter(list(y_pred), probabilities=proba, scores=scores, classes=classes)
    return Evaluator(EvaluationConfig(task=task, **config)).evaluate(dataset, adapter)
