"""The evaluator: validate, predict, measure, gate, compare.

Flow of :meth:`Evaluator.evaluate`:

1. Validate the dataset. Blocking errors end the evaluation with a failed report (no metrics).
2. Optionally exclude invalid rows (``invalid_rows='exclude'``), reporting counts and reasons.
3. Ask the model for predictions (read-only) and validate them. Blocking errors end it again.
4. Compute metrics. Undefined/unsupported ones are reported with a reason.
5. Apply threshold rules, baseline rules and the opt-in checks (robustness, repeatability,
   subgroups, custom checks).

Invalid *data or predictions* never raise; they become findings on the report. Invalid
*configuration* raises ``MLConfigurationError``. A model that raises while predicting raises
``ModelInvocationError``. The evaluator never trains or modifies a model or the dataset.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from complete_automation.core.context import utc_now
from complete_automation.core.serialization import dumps_deterministic
from complete_automation.ml.baseline import Baseline, compare_to_baseline
from complete_automation.ml.data import Dataset, is_numeric_column
from complete_automation.ml.errors import MLConfigurationError
from complete_automation.ml.metrics import (
    MetricContext,
    direction_of,
    get_metric,
    run_metric,
)
from complete_automation.ml.metrics.classification import (
    DEFAULT_LABEL_METRICS,
    DEFAULT_PROBABILITY_METRICS,
    DEFAULT_SCORE_METRICS,
    zero_division_notes,
)
from complete_automation.ml.metrics.clustering import EXTRINSIC_METRICS, INTRINSIC_METRICS
from complete_automation.ml.metrics.regression import DEFAULT_METRICS as REGRESSION_DEFAULTS
from complete_automation.ml.models import ModelAdapter, Predictions
from complete_automation.ml.robustness import max_difference, perturb, prediction_change
from complete_automation.ml.schemas import (
    BaselineComparison,
    BaselineRule,
    CheckResult,
    Direction,
    EvaluationConfig,
    EvaluationReport,
    GateSummary,
    Label,
    MetricStatus,
    MetricValue,
    PerturbationOutcome,
    PerturbationSpec,
    RepeatabilityResult,
    Severity,
    SubgroupResult,
    TaskType,
    ValidationFinding,
)
from complete_automation.ml.thresholds import evaluate_gates
from complete_automation.ml.validation import (
    align_columns,
    invalid_row_mask,
    resolve_classes,
    validate_dataset,
    validate_predictions,
)


class RepeatabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runs: int = Field(default=3, ge=2, le=20)
    stochastic: bool = False
    """Stochastic models need a tolerance; deterministic ones default to exact equality."""
    atol: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _tolerance(self) -> RepeatabilityConfig:
        if self.stochastic and self.atol <= 0:
            raise ValueError("a stochastic model needs atol > 0 (the expected variance)")
        return self


class SubgroupConfig(BaseModel):
    """Descriptive per-group metrics for an explicitly named column. Not a fairness verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    min_group_size: int = Field(default=30, ge=1)
    metrics: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CheckContext:
    dataset: Dataset
    predictions: Predictions
    config: EvaluationConfig
    metrics: tuple[MetricValue, ...]


class EvaluationCheck(Protocol):
    """Extension point for extra checks (fairness, explainability, domain rules) in later phases."""

    name: str

    def run(self, context: CheckContext) -> CheckResult: ...


@dataclass(frozen=True)
class _Prepared:
    dataset: Dataset
    predictions: Predictions
    classes: list[Label]
    y_true: np.ndarray | None


def _error(code: str, message: str, **details: JsonValue) -> ValidationFinding:
    return ValidationFinding(code=code, severity=Severity.ERROR, message=message, details=details)


def _warning(code: str, message: str, **details: JsonValue) -> ValidationFinding:
    return ValidationFinding(code=code, severity=Severity.WARNING, message=message, details=details)


class Evaluator:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        if config.metrics is not None:
            for name in config.metrics:
                get_metric(config.task, name)
        if config.task is TaskType.CLASSIFICATION and config.labels is not None and len(config.labels) < 2:
            raise MLConfigurationError("classification needs at least two labels")

    # --- public --------------------------------------------------------------------------------

    def evaluate(
        self,
        dataset: Dataset,
        model: ModelAdapter,
        *,
        baseline: Baseline | None = None,
        baseline_rules: Sequence[BaselineRule] = (),
        robustness: Sequence[PerturbationSpec] = (),
        repeatability: RepeatabilityConfig | None = None,
        subgroups: SubgroupConfig | None = None,
        checks: Sequence[EvaluationCheck] = (),
    ) -> EvaluationReport:
        config = self.config
        if baseline_rules and baseline is None:
            raise MLConfigurationError("baseline_rules were given without a baseline")
        if (robustness or repeatability) and not model.can_predict_new_inputs:
            raise MLConfigurationError("robustness and repeatability checks need a model that can predict, not precomputed predictions")

        findings = validate_dataset(dataset, config)
        if model.expected_rows is not None and model.expected_rows != len(dataset.frame):
            findings.append(_error(
                "prediction_length_mismatch",
                f"{model.expected_rows} prediction(s) for {len(dataset.frame)} row(s)",
                predictions=model.expected_rows, rows=len(dataset.frame)))
        if any(f.severity is Severity.ERROR for f in findings):
            return self._report(dataset, model, 0, 0, findings)

        working = dataset
        excluded = 0
        if config.invalid_rows == "exclude":
            mask, reasons = invalid_row_mask(dataset)
            if mask.any():
                excluded = int(mask.sum())
                working = dataset.subset(~mask)
                findings.append(_warning(
                    "rows_excluded", f"{excluded} row(s) excluded before evaluation",
                    count=excluded, reasons={k: v for k, v in reasons.items()}))
            if len(working.frame) == 0:
                findings.append(_error("empty_dataset", "every row was excluded as invalid"))
                return self._report(dataset, model, 0, excluded, findings)

        predictions = model.collect(working.X)
        y_true = working.y.to_numpy() if working.target is not None else None
        classes = (
            resolve_classes(y_true, predictions.labels, config)
            if config.task is TaskType.CLASSIFICATION
            else []
        )
        findings += validate_predictions(predictions, len(working.frame), classes, config)
        if any(f.severity is Severity.ERROR for f in findings):
            return self._report(dataset, model, len(working.frame), excluded, findings)

        prepared = _Prepared(working, predictions, classes, y_true)
        context = self._context(prepared, predictions, working.X)
        metrics = self._compute(context, predictions)
        findings += self._metric_findings(metrics, context)

        gates = evaluate_gates(config.thresholds, metrics, config.task) if config.thresholds else None
        comparison = None
        if baseline is not None:
            comparison = compare_to_baseline(baseline, metrics, baseline_rules, task=config.task)
            if baseline.dataset_fingerprint != dataset.metadata.fingerprint:
                findings.append(_warning(
                    "baseline_dataset_differs",
                    "the baseline was recorded on different data than this evaluation"))

        outcomes = tuple(self._robustness(prepared, model, spec, metrics) for spec in robustness)
        repeat = self._repeatability(prepared, model, repeatability, metrics) if repeatability else None
        groups: tuple[SubgroupResult, ...] = ()
        if subgroups is not None:
            groups, group_findings = self._subgroups(prepared, context, subgroups)
            findings += group_findings
        check_results = tuple(
            check.run(CheckContext(working, predictions, config, tuple(metrics))) for check in checks
        )
        return self._report(
            dataset, model, len(working.frame), excluded, findings, metrics=metrics, gates=gates,
            baseline=comparison, robustness=outcomes, repeatability=repeat, subgroups=groups,
            subgroup_column=subgroups.column if subgroups else None, checks=check_results,
            robustness_seeds=[s.seed for s in robustness],
        )

    # --- metrics -------------------------------------------------------------------------------

    def _context(self, prepared: _Prepared, predictions: Predictions, features: pd.DataFrame) -> MetricContext:
        config = self.config
        proba = scores = None
        if config.task is TaskType.CLASSIFICATION:
            if predictions.probabilities is not None:
                proba = align_columns(predictions.probabilities, predictions.classes, prepared.classes)
            if predictions.scores is not None:
                scores = align_columns(predictions.scores, predictions.classes, prepared.classes)
        y_true = prepared.y_true
        if config.task is TaskType.REGRESSION and y_true is not None:
            y_true = y_true.astype(float)
        matrix = None
        if config.task is TaskType.CLUSTERING and all(is_numeric_column(features[c]) for c in features):
            matrix = features.to_numpy(dtype=float)
        return MetricContext(config, y_true, predictions.labels, prepared.classes, proba, scores, matrix)

    def _metric_names(self, context: MetricContext) -> list[str]:
        if self.config.metrics is not None:
            return list(self.config.metrics)
        match self.config.task:
            case TaskType.CLASSIFICATION:
                names = list(DEFAULT_LABEL_METRICS)
                if context.probabilities is not None or context.scores is not None:
                    names += DEFAULT_SCORE_METRICS
                if context.probabilities is not None:
                    names += DEFAULT_PROBABILITY_METRICS
                return names
            case TaskType.REGRESSION:
                return list(REGRESSION_DEFAULTS)
            case _:
                return list(INTRINSIC_METRICS) + (list(EXTRINSIC_METRICS) if context.y_true is not None else [])

    def _compute(self, context: MetricContext, predictions: Predictions) -> list[MetricValue]:
        del predictions
        out: list[MetricValue] = []
        for name in self._metric_names(context):
            out += run_metric(get_metric(self.config.task, name), context)
        return out

    def _metric_findings(self, metrics: list[MetricValue], context: MetricContext) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        for item in metrics:
            if item.status is not MetricStatus.COMPUTED:
                findings.append(_warning(
                    f"metric_{item.status.value}", f"{item.name} is {item.status.value}: {item.reason}",
                    metric=item.name))
        if self.config.task is TaskType.CLASSIFICATION and self.config.zero_division == "zero":
            requested = {m.name for m in metrics}
            if requested & {"precision", "recall", "f1"} or any(n.startswith(("precision_", "recall_", "f1_")) for n in requested):
                notes = zero_division_notes(context)
                if notes:
                    findings.append(_warning(
                        "zero_division",
                        "precision/recall/F1 count 0/0 as 0 (zero_division='zero'): " + "; ".join(notes),
                        count=len(notes)))
        return findings

    # --- opt-in checks -------------------------------------------------------------------------

    def _robustness(
        self, prepared: _Prepared, model: ModelAdapter, spec: PerturbationSpec, original: list[MetricValue]
    ) -> PerturbationOutcome:
        config = self.config
        features = prepared.dataset.X
        perturbed, changed = perturb(features, spec)
        base = dict(name=spec.name, kind=spec.kind, features=spec.features, magnitude=spec.magnitude,
                    seed=spec.seed, cells_changed=changed)
        predictions = model.collect(perturbed)
        problems = validate_predictions(predictions, len(features), prepared.classes, config)
        if problems:
            return PerturbationOutcome(**base, prediction_change=None, passed=False,
                                       error="; ".join(p.message for p in problems if p.severity is Severity.ERROR))
        regression = config.task is TaskType.REGRESSION
        change = prediction_change(prepared.predictions.labels, predictions.labels, regression)
        changes: dict[str, dict[str, float]] = {}
        if prepared.y_true is not None:
            context = self._context(prepared, predictions, perturbed)
            before = {m.name: m for m in original if m.is_scalar}
            for item in self._compute(context, predictions):
                prior = before.get(item.name)
                if prior is None or not item.is_scalar:
                    continue
                direction = direction_of(config.task, item.name)
                if direction is Direction.NONE:
                    continue
                assert prior.value is not None and item.value is not None
                worse = prior.value - item.value if direction is Direction.HIGHER_IS_BETTER else item.value - prior.value
                changes[item.name] = {"original": prior.value, "perturbed": item.value, "degradation": worse}
        passed: bool | None = None
        error = None
        if spec.gate_metric is not None:
            entry = changes.get(spec.gate_metric)
            if entry is None:
                passed, error = False, f"gate metric {spec.gate_metric!r} is unavailable for this perturbation"
            else:
                passed = entry["degradation"] <= (spec.max_metric_drop or 0.0) + 1e-12
        if spec.max_prediction_change_rate is not None:
            within = change <= spec.max_prediction_change_rate + 1e-12
            passed = within if passed is None else passed and within
        return PerturbationOutcome(**base, prediction_change=change, metric_changes=changes, passed=passed, error=error)

    def _repeatability(
        self, prepared: _Prepared, model: ModelAdapter, repeat: RepeatabilityConfig, original: list[MetricValue]
    ) -> RepeatabilityResult:
        features = prepared.dataset.X
        first = {m.name: m.value for m in original if m.is_scalar}
        prediction_gap = metric_gap = 0.0
        for _ in range(repeat.runs - 1):
            again = model.collect(features)
            prediction_gap = max(
                prediction_gap,
                max_difference(prepared.predictions.labels, again.labels),
                max_difference(prepared.predictions.probabilities, again.probabilities),
            )
            context = self._context(prepared, again, features)
            for item in self._compute(context, again):
                if item.is_scalar and first.get(item.name) is not None:
                    metric_gap = max(metric_gap, abs(item.value - first[item.name]))  # type: ignore[operator]
        passed = prediction_gap <= repeat.atol and metric_gap <= repeat.atol
        return RepeatabilityResult(
            runs=repeat.runs, stochastic=repeat.stochastic, atol=repeat.atol,
            max_prediction_difference=prediction_gap if np.isfinite(prediction_gap) else 1e308,
            max_metric_difference=metric_gap, passed=passed)

    def _subgroups(
        self, prepared: _Prepared, context: MetricContext, spec: SubgroupConfig
    ) -> tuple[tuple[SubgroupResult, ...], list[ValidationFinding]]:
        config = self.config
        frame = prepared.dataset.frame
        if config.task is TaskType.CLUSTERING:
            raise MLConfigurationError("subgroup analysis supports classification and regression")
        if spec.column not in frame.columns:
            raise MLConfigurationError(f"subgroup column {spec.column!r} is not in the dataset")
        names = list(spec.metrics or (
            ("accuracy", "precision", "recall", "f1") if config.task is TaskType.CLASSIFICATION else ("mae", "rmse")))
        for name in names:
            get_metric(config.task, name)
        labels = frame[spec.column].astype("object").where(frame[spec.column].notna(), "<missing>").astype(str)
        distinct = sorted(labels.unique())
        if len(distinct) > config.max_groups:
            return (), [_error("too_many_groups", f"{spec.column!r} has {len(distinct)} distinct values; "
                               f"the limit is max_groups={config.max_groups} (is it a continuous column?)")]
        results: list[SubgroupResult] = []
        findings: list[ValidationFinding] = []
        for group in distinct:
            keep = (labels == group).to_numpy()
            size = int(keep.sum())
            if size < spec.min_group_size:
                results.append(SubgroupResult(group=group, n_samples=size, insufficient=True))
                findings.append(_warning(
                    "insufficient_subgroup", f"group {group!r} has {size} sample(s), fewer than {spec.min_group_size}",
                    group=group, count=size))
                continue
            sub = MetricContext(
                config, None if context.y_true is None else context.y_true[keep], context.y_pred[keep],
                context.classes,
                None if context.probabilities is None else context.probabilities[keep],
                None if context.scores is None else context.scores[keep], None)
            values: dict[str, float | None] = {}
            for name in names:
                for item in run_metric(get_metric(config.task, name), sub):
                    values[item.name] = item.value
            results.append(SubgroupResult(group=group, n_samples=size, insufficient=False, metrics=values))
        return tuple(results), findings

    # --- report --------------------------------------------------------------------------------

    def _report(
        self,
        dataset: Dataset,
        model: ModelAdapter,
        n_samples: int,
        n_excluded: int,
        findings: list[ValidationFinding],
        *,
        metrics: Sequence[MetricValue] = (),
        gates: GateSummary | None = None,
        baseline: BaselineComparison | None = None,
        robustness: tuple[PerturbationOutcome, ...] = (),
        repeatability: RepeatabilityResult | None = None,
        subgroups: tuple[SubgroupResult, ...] = (),
        subgroup_column: str | None = None,
        checks: tuple[CheckResult, ...] = (),
        robustness_seeds: Sequence[int] = (),
    ) -> EvaluationReport:
        config = self.config
        reasons = [f.message for f in findings if f.severity is Severity.ERROR]
        if gates is not None:
            reasons += [f"threshold failed: {o.metric} {o.reason}" for o in gates.threshold_failures]
            reasons += [f"metric unavailable: {o.metric} ({o.reason})" for o in gates.unavailable]
        if baseline is not None:
            reasons += [f"baseline regression: {c.metric}: {c.reason}" for c in baseline.comparisons if c.outcome == "regressed"]
            reasons += [f"baseline unavailable: {c.metric}: {c.reason}" for c in baseline.comparisons if c.outcome == "unavailable"]
        reasons += [f"robustness {o.name}: {o.error or 'limit exceeded'}" for o in robustness if o.passed is False]
        if repeatability is not None and not repeatability.passed:
            reasons.append("repeatability: predictions or metrics differ between runs beyond the tolerance")
        reasons += [f"check {c.name}: {c.summary}" for c in checks if c.passed is False]
        if config.on_undefined == "fail":
            reasons += [f"metric {m.name} is {m.status.value}: {m.reason}" for m in metrics if m.status is not MetricStatus.COMPUTED]
        if gates is not None and gates.warnings:
            findings = [*findings, *(_warning("gate_direction", w) for w in gates.warnings)]

        config_json = dumps_deterministic(config)
        identity = "|".join([config_json, dataset.metadata.fingerprint, model.info.model_id, model.info.version or ""])
        return EvaluationReport(
            evaluation_id=hashlib.sha256(identity.encode()).hexdigest()[:12],
            evaluated_at=utc_now(),
            task=config.task,
            model=model.info,
            dataset=dataset.metadata,
            n_samples=n_samples,
            n_excluded=n_excluded,
            config=config,
            metrics=tuple(metrics),
            gates=gates,
            findings=tuple(findings),
            baseline=baseline,
            robustness=robustness,
            repeatability=repeatability,
            subgroups=subgroups,
            subgroup_column=subgroup_column,
            checks=checks,
            reproducibility=_reproducibility(config_json, dataset, config.seed, robustness_seeds),
            passed=not reasons,
            failure_reasons=tuple(reasons),
        )


def _reproducibility(config_json: str, dataset: Dataset, seed: int, seeds: Sequence[int]) -> dict[str, JsonValue]:
    import importlib.metadata as metadata

    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "unknown"

    return {
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "dataset_fingerprint": dataset.metadata.fingerprint,
        "seed": seed,
        "perturbation_seeds": list(seeds),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
    }


__all__ = ["CheckContext", "EvaluationCheck", "Evaluator", "RepeatabilityConfig", "SubgroupConfig"]
