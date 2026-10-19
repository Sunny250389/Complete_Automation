"""Typed, immutable contracts of the LLM evaluation module.

Raw data and verdicts are separate on purpose: a :class:`ModelResponse` is what the model returned
(captured once), a :class:`MetricOutcome` is one evaluator's verdict on it, and an
:class:`EvaluationRecord` ties the two together. The same captured response can therefore be
evaluated again, by other metrics, without calling the model.

Metadata a provider did not expose is ``None`` ("unavailable"); nothing is estimated.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from complete_automation.core.artifacts import ArtifactRef
from complete_automation.core.context import utc_now

SCHEMA_VERSION = "1.0"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


# ---------------------------------------------------------------- configuration


class GenerationConfig(_Model):
    """Sampling settings. ``None`` means "not sent": the server default applies and is not recorded
    as a value, so a report never claims a setting that was not made."""

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    """Output token limit (Ollama ``num_predict``)."""
    seed: int | None = None
    stop: tuple[str, ...] = ()
    think: bool | None = None
    """Reasoning models: ``False`` disables thinking, ``True`` enables it, ``None`` leaves it as is."""


class ModelConfig(_Model):
    provider: str = "ollama"
    model: str = Field(min_length=1)
    base_url: str = "http://localhost:11434"
    mode: Literal["chat", "generate"] = "chat"
    stream: bool = False
    timeout_seconds: float = Field(default=120.0, gt=0)
    generation: GenerationConfig = GenerationConfig()


class PromptTemplate(_Model):
    """A versioned prompt. Placeholders are ``{prompt}`` and ``{context}`` (``str.format`` syntax).

    Only the case prompt and context are available to a template; the reference answer is never
    sent to the model under test.
    """

    template_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    system: str | None = None
    user: str = "{prompt}"

    def render(self, prompt: str, context: tuple[str, ...] = ()) -> tuple[str | None, str]:
        numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context, start=1))
        try:
            return self.system, self.user.format(prompt=prompt, context=numbered)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                f"template {self.label} uses an unknown placeholder or bad braces ({error}); "
                "available: {prompt}, {context}"
            ) from None

    @property
    def label(self) -> str:
        return f"{self.template_id}@{self.version}"


RAG_TEMPLATE = PromptTemplate(
    template_id="rag-basic",
    version="1",
    system="Answer the question using only the numbered context. "
    "If the context does not contain the answer, say that you cannot answer from the context.",
    user="Context:\n{context}\n\nQuestion: {prompt}",
)


class JudgeConfig(_Model):
    model: ModelConfig
    """The judge is configured independently of the model under test."""
    scale_min: float = 1.0
    scale_max: float = 5.0
    repeats: int = Field(default=1, ge=1, le=10)
    """Judge calls per criterion. Above 1 the spread is reported, not hidden in an average."""

    @model_validator(mode="after")
    def _scale(self) -> JudgeConfig:
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        return self


# ---------------------------------------------------------------- test cases

CheckType = Literal[
    "not_empty",
    "exact_match",
    "contains",
    "not_contains",
    "regex",
    "json_valid",
    "json_schema",
    "json_keys",
    "length",
    "refusal",
    "completed",
    "context_overlap",
]


class CheckSpec(_Model):
    """One deterministic check. Which fields apply depends on ``type`` (see ``docs/llm_evaluation.md``)."""

    type: CheckType
    name: str | None = None
    value: Any = None
    """exact_match: expected text (default: the case reference). contains/not_contains: a string or a
    list of strings. regex: a pattern. json_schema: a JSON Schema (default: the case
    ``expected_schema``). json_keys: required keys. refusal: ``true`` (must refuse) or ``false``
    (must not refuse)."""
    case_sensitive: bool = True
    normalize_whitespace: bool = False
    """exact_match: collapse whitespace runs and strip the ends before comparing."""
    allowed_keys: tuple[str, ...] | None = None
    """json_keys: when set, any other top-level key is unexpected and fails the check."""
    minimum: float | None = None
    maximum: float | None = None
    unit: Literal["chars", "words"] = "chars"
    patterns: tuple[str, ...] = ()
    """refusal: regular expressions that count as a refusal (default list in ``evaluators.deterministic``)."""

    @property
    def label(self) -> str:
        return self.name or self.type


class CriterionSpec(_Model):
    """One judge-evaluated criterion."""

    name: str = Field(min_length=1)
    """A built-in criterion (see ``evaluators.rubrics``) or a custom name together with ``rubric``."""
    rubric: str | None = None
    rubric_version: str | None = None
    min_score: float | None = None
    """Pass threshold on the judge scale. ``None`` records the score without a verdict."""


OPERATOR_SYMBOLS = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "=="}


class ThresholdRule(_Model):
    """``metric <operator> value``, for case-level and suite-level gates."""

    metric: str = Field(min_length=1)
    operator: Literal["ge", "gt", "le", "lt", "eq"]
    value: float
    tolerance: float = Field(default=0.0, ge=0)
    """Only for ``eq``."""
    required: bool = True

    @property
    def expression(self) -> str:
        return f"{self.metric} {OPERATOR_SYMBOLS[self.operator]} {self.value:g}"


class TestCase(_Model):
    __test__ = False  # not a pytest test class

    case_id: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    reference: str | None = None
    expected_schema: dict[str, Any] | None = None
    context: tuple[str, ...] = ()
    checks: tuple[CheckSpec, ...] = ()
    criteria: tuple[CriterionSpec, ...] = ()
    thresholds: tuple[ThresholdRule, ...] = ()
    """Gates on this case's numeric metrics, for example ``latency_seconds``."""
    tags: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("case_id")
    @classmethod
    def _id_chars(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value):
            raise ValueError("case_id may use letters, digits and . _ : - only")
        return value

    @field_validator("context", mode="before")
    @classmethod
    def _context_list(cls, value: Any) -> Any:
        return (value,) if isinstance(value, str) else value


class Dataset(_Model):
    dataset_id: str = Field(min_length=1)
    version: str = "unversioned"
    fingerprint: str
    cases: tuple[TestCase, ...]


# ---------------------------------------------------------------- model request / response


class Message(_Model):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(_Model):
    messages: tuple[Message, ...] = Field(min_length=1)
    config: ModelConfig
    response_format: Literal["json"] | dict[str, Any] | None = None
    """Ask the provider for JSON output (``"json"``) or output conforming to a JSON Schema."""
    template: str | None = None
    """``template_id@version`` when a prompt template produced the messages."""


class RequestStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    """A stream ended before the provider signalled completion; ``text`` is what arrived."""
    FAILED = "failed"
    TIMEOUT = "timeout"


class ErrorInfo(_Model):
    kind: str
    message: str
    status_code: int | None = None


class ModelResponse(_Model):
    """What a model call returned, or how it failed. Captured once; never edited."""

    provider: str
    model: str
    model_identifier: str | None = None
    """Version or digest, when the provider exposes one. ``None`` means unavailable."""
    request: ModelRequest
    status: RequestStatus
    text: str = ""
    thinking: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_seconds: float = Field(ge=0)
    timestamp: datetime = Field(default_factory=utc_now)
    error: ErrorInfo | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is RequestStatus.COMPLETED


# ---------------------------------------------------------------- verdicts


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    """The evaluator could not reach a verdict. Never counted as a pass."""
    SCORED = "scored"
    """Informational measurement, no pass/fail rule attached."""
    NOT_EVALUATED = "not_evaluated"
    """No response existed to evaluate."""


class MetricOutcome(_Model):
    name: str
    kind: Literal["deterministic", "judge", "measurement"]
    outcome: Outcome
    value: float | None = None
    threshold: float | None = None
    explanation: str = ""
    details: dict[str, JsonValue] = Field(default_factory=dict)


class JudgeResult(_Model):
    criterion: str
    score: float | None = None
    scale_min: float
    scale_max: float
    verdict: Literal["pass", "fail", "uncertain"] | None = None
    rationale: str = ""
    evidence: tuple[str, ...] = ()
    judge_provider: str
    judge_model: str
    rubric_version: str
    scores: tuple[float, ...] = ()
    """One score per repeat when repeats > 1."""
    error: str | None = None
    raw_output: str | None = None
    latency_seconds: float = 0.0

    @property
    def usable(self) -> bool:
        return self.error is None and self.score is not None


class GateOutcome(_Model):
    scope: Literal["case", "suite"]
    metric: str
    operator: str
    threshold: float
    observed: float | None
    kind: Literal["passed", "threshold_failed", "metric_unavailable"]
    required: bool = True
    reason: str


class RecordStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    """A check, judged criterion or threshold was not met: the model's output is at fault."""
    MODEL_ERROR = "model_error"
    """The model call failed or was incomplete: no (complete) response to evaluate."""
    EVALUATOR_ERROR = "evaluator_error"
    """A response exists but an evaluator could not reach a verdict."""


class EvaluationRecord(_Model):
    record_id: str
    case: TestCase
    response: ModelResponse
    metrics: tuple[MetricOutcome, ...] = ()
    judge_results: tuple[JudgeResult, ...] = ()
    gates: tuple[GateOutcome, ...] = ()
    status: RecordStatus
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class SuiteGateResult(_Model):
    passed: bool
    outcomes: tuple[GateOutcome, ...] = ()
    warnings: tuple[str, ...] = ()


class BaselineRule(_Model):
    """Allowed worsening against a baseline; ``direction`` says which way is better."""

    metric: str
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    max_drop: float = Field(default=0.0, ge=0)
    required: bool = True


class MetricComparison(_Model):
    metric: str
    baseline: float | None
    current: float | None
    difference: float | None
    outcome: Literal["unchanged", "changed", "passed", "regressed", "unavailable"]
    reason: str | None = None


class BaselineComparison(_Model):
    baseline_id: str
    comparisons: tuple[MetricComparison, ...]
    passed: bool


class EvaluationConfig(_Model):
    """Everything that defines a run; recorded verbatim in its report."""

    model: ModelConfig
    judge: JudgeConfig | None = None
    template: PromptTemplate | None = None
    suite_thresholds: tuple[ThresholdRule, ...] = ()
    baseline_rules: tuple[BaselineRule, ...] = ()
    evaluate_partial: bool = False
    """Evaluate a response that ended mid-stream. Off by default: partial output is a model error."""
    default_system_prompt: str | None = None


class SuiteReport(_Model):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str
    evaluation_id: str
    started_at: datetime
    finished_at: datetime
    dataset_id: str
    dataset_fingerprint: str | None = None
    config: EvaluationConfig
    records: tuple[EvaluationRecord, ...]
    aggregates: dict[str, float | None]
    counts: dict[str, int]
    gates: SuiteGateResult
    baseline: BaselineComparison | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            bool(self.records)
            and all(r.status is RecordStatus.PASSED for r in self.records)
            and self.gates.passed
            and (self.baseline is None or self.baseline.passed)
        )

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons = [
            f"{r.case.case_id} [{r.status.value}]: {'; '.join(r.reasons)}"
            for r in self.records
            if r.status is not RecordStatus.PASSED
        ]
        reasons += [
            f"suite gate {o.metric}: {o.reason}"
            for o in self.gates.outcomes
            if o.required and o.kind != "passed"
        ]
        if self.baseline is not None:
            reasons += [
                f"baseline {c.metric}: {c.reason}"
                for c in self.baseline.comparisons
                if c.outcome in ("regressed", "unavailable")
            ]
        if not self.records:
            reasons.append("the evaluation contained no test cases")
        return tuple(reasons)

    def assert_passed(self) -> None:
        """Raise the error matching the cause; model and evaluator problems come before thresholds."""
        from complete_automation.llm_evaluation.errors import (
            EvaluatorError,
            ModelInvocationError,
            QualityGateError,
        )

        if self.passed:
            return
        body = "\n  ".join(self.failure_reasons)
        statuses = {r.status for r in self.records}
        if RecordStatus.MODEL_ERROR in statuses:
            raise ModelInvocationError(f"model invocation failed or was incomplete:\n  {body}")
        if RecordStatus.EVALUATOR_ERROR in statuses:
            raise EvaluatorError(f"an evaluator could not reach a verdict:\n  {body}")
        raise QualityGateError(f"LLM evaluation failed:\n  {body}")
