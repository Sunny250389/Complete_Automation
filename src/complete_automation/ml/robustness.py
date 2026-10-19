"""Opt-in robustness checks: controlled input perturbations and repeatability.

These probe sensitivity to small, explicitly configured input changes. They make no claim about
adversarial robustness or production reliability, and a changed prediction is reported, not
assumed to be a defect. The original features are never modified: every perturbation works on a
copy and is seeded, so a run is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from complete_automation.ml.data import is_numeric_column
from complete_automation.ml.errors import MLConfigurationError
from complete_automation.ml.schemas import PerturbationSpec


def perturb(features: pd.DataFrame, spec: PerturbationSpec) -> tuple[pd.DataFrame, int]:
    """Return a perturbed copy of ``features`` and the number of cells that changed."""
    missing = [c for c in spec.features if c not in features.columns]
    if missing:
        raise MLConfigurationError(f"perturbation {spec.name!r}: unknown feature(s) {missing}")
    out = features.copy(deep=True)
    rng = np.random.default_rng(spec.seed)
    changed = 0
    for column in spec.features:
        original = features[column]
        if spec.kind == "substitute_category":
            assert spec.mapping is not None
            out[column] = original.map(lambda v: spec.mapping.get(v, v) if isinstance(v, str) else v)
        elif spec.kind == "set_missing":
            hit = rng.random(len(out)) < spec.magnitude
            out[column] = original.astype("float64" if is_numeric_column(original) else "object")
            out.loc[hit, column] = np.nan
        else:
            if not is_numeric_column(original):
                raise MLConfigurationError(
                    f"perturbation {spec.name!r}: {spec.kind} needs numeric feature, {column!r} is not"
                )
            values = original.to_numpy(dtype=float)
            if spec.kind == "gaussian_noise":
                spread = float(np.nanstd(values))
                out[column] = values + rng.normal(0.0, spec.magnitude * spread, len(values))
            else:
                out[column] = values * (1.0 + rng.uniform(-spec.magnitude, spec.magnitude, len(values)))
        changed += int(_differs(original, out[column]).sum())
    return out, changed


def _differs(before: pd.Series, after: pd.Series) -> np.ndarray:
    equal = (before == after) | (before.isna() & after.isna())
    return ~equal.to_numpy(dtype=bool)


def prediction_change(original: np.ndarray, perturbed: np.ndarray, regression: bool) -> float:
    """Share of changed labels, or mean absolute change for numeric predictions."""
    if regression:
        return float(np.mean(np.abs(perturbed.astype(float) - original.astype(float))))
    return float(np.mean(np.asarray(original != perturbed, dtype=bool)))


def max_difference(first: np.ndarray | None, second: np.ndarray | None) -> float:
    """Largest absolute difference between two prediction arrays; 1.0 if labels differ, 0.0 if equal."""
    if first is None or second is None:
        return 0.0
    if first.shape != second.shape:
        return float("inf")
    if np.issubdtype(first.dtype, np.number) and np.issubdtype(second.dtype, np.number):
        return float(np.max(np.abs(first.astype(float) - second.astype(float)))) if first.size else 0.0
    return float(np.any(np.asarray(first != second, dtype=bool)))
