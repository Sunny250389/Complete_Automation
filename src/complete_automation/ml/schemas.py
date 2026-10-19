"""Typed contracts of the ML evaluation module.

Evaluation results are separate from whatever a model returns: they hold metric values,
validation findings, gate outcomes and metadata, never raw rows or raw predictions.
All models are immutable and JSON-serialisable.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from complete_automation.core.artifacts import ArtifactRef
from complete_automation.ml.errors import (
    MetricUnavailableError,
    QualityGateError,
    ValidationFailedError,
)

SCHEMA_VERSION = "1.0"

Label = str | int | bool
"""A class label. Labels are compared by value and type, so ``1`` and ``"1"`` differ."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class Severity(StrEnum):
    ERROR = "error"
    """Blocking: metrics are not computed, the report fails."""
    WARNING = "warning"
    INFO = "info"


class MetricStatus(StrEnum):
    COMPUTED = "computed"
    UNDEFINED = "undefined"
    """The data make the metric mathematically undefined (one class, zero targets, ...)."""
    UNSUPPORTED = "unsupported"
    """The task or the available model outputs cannot provide this metric."""


class Direction(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NONE = "none"
    """Not an ordered quality score (confusion matrix, support counts)."""


class Operator(StrEnum):
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    EQ = "eq"


OPERATOR_SYMBOLS = {
    Operator.GE: ">=",
    Operator.GT: ">",
    Operator.LE: "<=",
    Operator.LT: "<",
    Operator.EQ: "==",
}


# --- dataset contract -------------------------------------------------------------------------


class ColumnSpec(_Model):
    dtype: Literal["any", "numeric", "integer", "string", "boolean"] = "any"
    nullable: bool = False
    allowed_values: tuple[Label, ...] | None = None
    min: float | None = None
    max: float | None = None


class DatasetSchema(_Model):
    """Expected columns of a dataset. Only the declared columns are checked for their type."""

    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    forbid_extra_columns: bool = False
    unique_key: tuple[str, ...] | None = None
    """Columns that must identify a row uniquely; whole-row duplicates are checked otherwise."""
    duplicates: Literal["off", "warn", "error"] = "warn"


class DatasetMetadata(_Model):
    dataset_id: str = Field(min_length=1)
    version: str | None = None
    source: str | None = None
    """Path relative to the data root, never an absolute location."""
    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
    fingerprint: str
    """SHA-256 of the column names and cell hashes; identifies the data without storing them."""


class ModelInfo(_Model):
    model_id: str = Field(min_length=1)
    version: str | None = None
    kind: str = "unspecified"
    """``precomputed``, ``estimator``, ``function`` or an adapter-defined value."""
    class_name: str | None = None


# --- metrics, gates ---------------------------------------------------------------------------


class MetricValue(_Model):
    name: str = Field(min_length=1)
    status: MetricStatus
    value: float | None = None
    """Set for scalar, computed metrics. ``None`` for undefined, unsupported and structured ones."""
    direction: Direction = Direction.NONE
    reason: str | None = None
    detail: JsonValue | None = None

    @model_validator(mode="after")
    def _consistent(self) -> MetricValue:
        if self.status is not MetricStatus.COMPUTED:
            if self.value is not None:
                raise ValueError("only a computed metric carries a value")
            if not self.reason:
                raise ValueError(f"a {self.status.value} metric needs a reason")
        return self

    @property
    def is_scalar(self) -> bool:
        return self.status is MetricStatus.COMPUTED and self.value is not None


class ThresholdRule(_Model):
    """``<metric> <operator> <value>``. Every rule is explicit; there are no implicit defaults."""

    metric: str = Field(min_length=1)
    operator: Operator
    value: float
    tolerance: float = Field(default=0.0, ge=0.0)
    """Absolute tolerance, used by ``eq`` only."""
    required: bool = True
    """A non-required rule is reported but does not fail the gate."""

    @classmethod
    def at_least(cls, metric: str, value: float, **kw: Any) -> ThresholdRule:
        return cls(metric=metric, operator=Operator.GE, value=value, **kw)

    @classmethod
    def at_most(cls, metric: str, value: float, **kw: Any) -> ThresholdRule:
        return cls(metric=metric, operator=Operator.LE, value=value, **kw)

    @classmethod
    def above(cls, metric: str, value: float, **kw: Any) -> ThresholdRule:
        return cls(metric=metric, operator=Operator.GT, value=value, **kw)

    @classmethod
    def below(cls, metric: str, value: float, **kw: Any) -> ThresholdRule:
        return cls(metric=metric, operator=Operator.LT, value=value, **kw)

    @classmethod
    def equals(cls, metric: str, value: float, tolerance: float = 0.0, **kw: Any) -> ThresholdRule:
        return cls(metric=metric, operator=Operator.EQ, value=value, tolerance=tolerance, **kw)

    @model_validator(mode="after")
    def _tolerance_only_for_eq(self) -> ThresholdRule:
        if self.tolerance and self.operator is not Operator.EQ:
            raise ValueError("tolerance applies to the eq operator only")
        return self

    @property
    def expression(self) -> str:
        return f"{self.metric} {OPERATOR_SYMBOLS[self.operator]} {self.value:.10g}"


GateKind = Literal["passed", "threshold_failed", "metric_unavailable"]


class GateOutcome(_Model):
    metric: str
    operator: Operator
    threshold: float
    observed: float | None
    kind: GateKind
    required: bool = True
    reason: str

    @property
    def passed(self) -> bool:
        return self.kind == "passed"


class GateSummary(_Model):
    passed: bool
    outcomes: tuple[GateOutcome, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def threshold_failures(self) -> tuple[GateOutcome, ...]:
        return tuple(o for o in self.outcomes if o.required and o.kind == "threshold_failed")

    @property
    def unavailable(self) -> tuple[GateOutcome, ...]:
        return tuple(o for o in self.outcomes if o.required and o.kind == "metric_unavailable")


# --- baseline ----------------------------------------------------------------------------------


class BaselineRule(_Model):
    """How much worse than the baseline a metric may get. Direction comes from the metric."""

    metric: str = Field(min_length=1)
    max_drop: float | None = Field(default=None, ge=0.0)
    """Largest allowed absolute worsening, in metric units."""
    max_relative_drop: float | None = Field(default=None, ge=0.0)
    """Largest allowed worsening as a fraction of ``abs(baseline)``."""
    required: bool = True

    @model_validator(mode="after")
    def _has_limit(self) -> BaselineRule:
        if self.max_drop is None and self.max_relative_drop is None:
            raise ValueError("a baseline rule needs max_drop or max_relative_drop")
        return self


class MetricComparison(_Model):
    metric: str
    baseline: float | None
    current: float | None
    absolute_difference: float | None
    """``current - baseline``."""
    relative_difference: float | None
    """``(current - baseline) / abs(baseline)``; ``None`` when the baseline is zero or undefined."""
    outcome: Literal["unchanged", "changed", "passed", "regressed", "unavailable"]
    """``passed``/``regressed``/``unavailable`` appear only for metrics with a baseline rule."""
    reason: str | None = None


class BaselineComparison(_Model):
    baseline_id: str
    comparisons: tuple[MetricComparison, ...]
    passed: bool


# --- robustness, repeatability, subgroups, extra checks ---------------------------------------


class PerturbationSpec(_Model):
    name: str = Field(min_length=1)
    kind: Literal["gaussian_noise", "relative_scale", "set_missing", "substitute_category"]
    features: tuple[str, ...] = Field(min_length=1)
    magnitude: float = Field(default=0.0, ge=0.0, le=1.0)
    """``gaussian_noise``: noise std as a fraction of each feature's std.
    ``relative_scale``: each value is multiplied by ``1 + u``, ``u`` uniform in ±magnitude.
    ``set_missing``: fraction of rows whose value becomes missing."""
    mapping: dict[str, str] | None = None
    """``substitute_category``: replacement per original value; other values are kept."""
    seed: int = 0
    allow_missing_inputs: bool = False
    """Must be ``True`` for ``set_missing``: the model's input contract has to allow missing values."""
    gate_metric: str | None = None
    max_metric_drop: float | None = Field(default=None, ge=0.0)
    max_prediction_change_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> PerturbationSpec:
        if self.kind == "substitute_category":
            if not self.mapping:
                raise ValueError("substitute_category needs a mapping")
        elif self.magnitude <= 0:
            raise ValueError(f"{self.kind} needs a magnitude greater than 0")
        if self.kind == "set_missing" and not self.allow_missing_inputs:
            raise ValueError("set_missing needs allow_missing_inputs=True (the model must accept it)")
        if (self.max_metric_drop is None) != (self.gate_metric is None):
            raise ValueError("gate_metric and max_metric_drop go together")
        return self


class PerturbationOutcome(_Model):
    name: str
    kind: str
    features: tuple[str, ...]
    magnitude: float
    seed: int
    cells_changed: int
    prediction_change: float | None
    """Classification and clustering: share of changed labels. Regression: mean absolute change."""
    metric_changes: dict[str, dict[str, float]] = Field(default_factory=dict)
    """metric -> ``original``, ``perturbed``, ``degradation`` (positive means worse)."""
    passed: bool | None = None
    error: str | None = None


class RepeatabilityResult(_Model):
    runs: int
    stochastic: bool
    atol: float
    max_prediction_difference: float
    max_metric_difference: float
    passed: bool


class SubgroupResult(_Model):
    group: str
    n_samples: int
    insufficient: bool
    metrics: dict[str, float | None] = Field(default_factory=dict)


class CheckResult(_Model):
    """Result of a user-supplied :class:`~complete_automation.ml.evaluator.EvaluationCheck`."""

    name: str
    passed: bool | None
    summary: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ValidationFinding(_Model):
    code: str
    severity: Severity
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)
    """Counts and column names only; offending values are never recorded."""


# --- configuration ----------------------------------------------------------------------------


class EvaluationConfig(_Model):
    task: TaskType
    metrics: tuple[str, ...] | None = None
    """``None`` selects the task's default metrics (see docs/ml_evaluation.md)."""
    thresholds: tuple[ThresholdRule, ...] = ()
    labels: tuple[Label, ...] | None = None
    """The full set of valid classes. When set, other labels in targets or predictions are errors."""
    positive_label: Label | None = None
    averaging: Literal["binary", "macro", "micro", "weighted"] = "macro"
    """Averaging for ``precision``, ``recall`` and ``f1``; ``binary`` needs ``positive_label``."""
    zero_division: Literal["zero", "undefined"] = "zero"
    """``zero``: a 0/0 precision, recall or F1 counts as 0 and a warning lists it.
    ``undefined``: the metric becomes undefined instead."""
    probability_atol: float = Field(default=1e-6, gt=0)
    mape_zero_policy: Literal["undefined", "exclude"] = "undefined"
    mape_epsilon: float = Field(default=1e-12, ge=0)
    """Targets with ``abs(y) <= mape_epsilon`` are treated as zero for MAPE."""
    noise_label: int | None = None
    """Clustering: label of noise points (for example -1 for DBSCAN), left out of intrinsic metrics."""
    invalid_rows: Literal["error", "exclude"] = "error"
    """``exclude`` drops rows with null targets, null features or non-finite numbers and reports why."""
    on_undefined: Literal["warn", "fail"] = "warn"
    """What a requested metric that is undefined or unsupported does to the report."""
    imbalance_warning_ratio: float = Field(default=5.0, gt=1.0)
    seed: int = 0
    max_groups: int = Field(default=50, ge=1)

    @field_validator("metrics")
    @classmethod
    def _no_duplicate_metrics(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("metrics contains duplicates")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> EvaluationConfig:
        if self.labels is not None:
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("labels contains duplicates")
            if self.positive_label is not None and self.positive_label not in self.labels:
                raise ValueError("positive_label is not one of labels")
        if self.averaging == "binary" and self.positive_label is None:
            raise ValueError("averaging='binary' needs positive_label")
        return self


# --- report -----------------------------------------------------------------------------------


class EvaluationReport(_Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    evaluation_id: str
    evaluated_at: datetime
    task: TaskType
    model: ModelInfo
    dataset: DatasetMetadata
    n_samples: int = Field(ge=0)
    n_excluded: int = Field(default=0, ge=0)
    config: EvaluationConfig
    metrics: tuple[MetricValue, ...] = ()
    gates: GateSummary | None = None
    findings: tuple[ValidationFinding, ...] = ()
    baseline: BaselineComparison | None = None
    robustness: tuple[PerturbationOutcome, ...] = ()
    repeatability: RepeatabilityResult | None = None
    subgroups: tuple[SubgroupResult, ...] = ()
    subgroup_column: str | None = None
    checks: tuple[CheckResult, ...] = ()
    reproducibility: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    passed: bool
    failure_reasons: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    def metric(self, name: str) -> MetricValue:
        for item in self.metrics:
            if item.name == name:
                return item
        raise KeyError(f"metric {name!r} was not computed; available: {[m.name for m in self.metrics]}")

    def value(self, name: str) -> float:
        item = self.metric(name)
        if item.value is None:
            raise MetricUnavailableError(f"metric {name!r} is {item.status.value}: {item.reason}")
        return item.value

    def assert_passed(self) -> None:
        """Raise the error matching the cause of a failed report; do nothing when it passed."""
        if self.passed:
            return
        message = "ML evaluation failed:\n" + "\n".join(f"  - {r}" for r in self.failure_reasons)
        if self.errors:
            raise ValidationFailedError(message)
        if self.gates is not None and self.gates.threshold_failures:
            raise QualityGateError(message)
        if self.baseline is not None and any(c.outcome == "regressed" for c in self.baseline.comparisons):
            raise QualityGateError(message)
        if (self.gates is not None and self.gates.unavailable) or any(
            m.status is not MetricStatus.COMPUTED for m in self.metrics
        ):
            raise MetricUnavailableError(message)
        raise QualityGateError(message)
