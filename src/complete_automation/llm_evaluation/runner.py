"""The evaluation flow: build request -> call the model -> capture -> evaluate -> gate.

``invoke`` is the only place the model under test is called. ``evaluate_response`` works on an
already captured :class:`ModelResponse` and never calls it, so one response can be evaluated again
with other metrics. A model that fails to answer yields a ``MODEL_ERROR`` record: the framework
never substitutes a fixture or reference text for a missing response.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from complete_automation.core.context import utc_now
from complete_automation.llm_evaluation import gates
from complete_automation.llm_evaluation.clients.base import ModelClient
from complete_automation.llm_evaluation.errors import ModelInvocationError
from complete_automation.llm_evaluation.evaluators.deterministic import run_check
from complete_automation.llm_evaluation.evaluators.judge import JudgeEvaluator
from complete_automation.llm_evaluation.schemas import (
    Dataset,
    ErrorInfo,
    EvaluationConfig,
    EvaluationRecord,
    JudgeResult,
    Message,
    MetricOutcome,
    ModelRequest,
    ModelResponse,
    Outcome,
    RecordStatus,
    RequestStatus,
    SuiteReport,
    TestCase,
)

_MEASUREMENTS = ("latency_seconds", "output_tokens", "response_chars")


class EvaluationRunner:
    """Evaluates ``client``'s model on test cases.

    ``judge`` is optional and independent of ``client``: it is a different
    :class:`JudgeEvaluator` with its own client and model configuration.
    """

    def __init__(
        self,
        client: ModelClient,
        config: EvaluationConfig,
        *,
        judge: JudgeEvaluator | None = None,
        run_id: str = "run",
    ) -> None:
        if config.judge is not None and judge is None:
            raise ValueError("config.judge is set but no JudgeEvaluator was given")
        if judge is not None and config.judge is None:
            config = config.model_copy(update={"judge": judge.config})
        self.client = client
        self.config = config
        self.judge = judge
        self.run_id = run_id
        self._identifier: str | None | bool = False  # False = not looked up yet

    # -- step 1-2: request ------------------------------------------------------------------

    def build_request(self, case: TestCase) -> ModelRequest:
        template = self.config.template
        if template is not None:
            system, user = template.render(case.prompt, case.context)
        else:
            system, user = None, case.prompt
        system = case.system_prompt or system or self.config.default_system_prompt
        messages = ([Message(role="system", content=system)] if system else []) + [
            Message(role="user", content=user)
        ]
        return ModelRequest(
            messages=tuple(messages),
            config=self.config.model,
            template=template.label if template else None,
        )

    # -- step 3-4: call and capture ----------------------------------------------------------

    def invoke(self, case: TestCase) -> ModelResponse:
        """Call the configured model once. Failures are captured, not raised."""
        request = self.build_request(case)
        try:
            response = self.client.generate(request)
        except ModelInvocationError as error:
            return ModelResponse(
                provider=self.config.model.provider,
                model=self.config.model.model,
                request=request,
                status=RequestStatus.TIMEOUT if error.error_kind == "timeout" else RequestStatus.FAILED,
                latency_seconds=0.0,
                error=ErrorInfo(
                    kind=error.error_kind, message=str(error), status_code=getattr(error, "status_code", None)
                ),
            )
        if response.model_identifier is None:
            response = response.model_copy(update={"model_identifier": self._model_identifier()})
        return response

    def _model_identifier(self) -> str | None:
        if self._identifier is False:
            try:
                self._identifier = self.client.identify(self.config.model)
            except ModelInvocationError:
                self._identifier = None
        return self._identifier  # type: ignore[return-value]

    # -- step 5-6: evaluate a captured response ---------------------------------------------------

    def evaluate_response(self, case: TestCase, response: ModelResponse) -> EvaluationRecord:
        """Evaluate a captured response. Never calls the model under test."""
        record_id = f"{case.case_id}-{uuid.uuid4().hex[:8]}"
        evaluable = response.ok or (self.config.evaluate_partial and response.status is RequestStatus.PARTIAL)
        if not evaluable:
            return self._unevaluated(record_id, case, response)

        metrics: list[MetricOutcome] = [
            MetricOutcome(name="response_chars", kind="measurement", outcome=Outcome.SCORED,
                          value=float(len(response.text))),
            MetricOutcome(name="latency_seconds", kind="measurement", outcome=Outcome.SCORED,
                          value=response.latency_seconds),
        ]
        if response.output_tokens is not None:
            metrics.append(MetricOutcome(name="output_tokens", kind="measurement",
                                         outcome=Outcome.SCORED, value=float(response.output_tokens)))
        metrics += [run_check(spec, case, response) for spec in case.checks]
        judge_results: list[JudgeResult] = []
        for criterion in case.criteria:
            if self.judge is None:
                metrics.append(MetricOutcome(
                    name=f"judge.{criterion.name}", kind="judge", outcome=Outcome.ERROR,
                    explanation="the case asks for a judged criterion but no judge is configured"))
                continue
            outcome, result = self.judge.evaluate(case, response, criterion)
            metrics.append(outcome)
            judge_results.append(result)

        warnings = []
        if response.status is RequestStatus.PARTIAL:
            warnings.append("response is partial; evaluated because evaluate_partial is enabled")
        case_gates = gates.evaluate_rules(case.thresholds, gates.record_values(metrics), "case")
        status, reasons = _status(metrics, case_gates)
        return EvaluationRecord(
            record_id=record_id, case=case, response=response, metrics=tuple(metrics),
            judge_results=tuple(judge_results), gates=case_gates, status=status,
            reasons=tuple(reasons), warnings=tuple(warnings),
        )

    def _unevaluated(self, record_id: str, case: TestCase, response: ModelResponse) -> EvaluationRecord:
        why = response.error.message if response.error else f"request status {response.status.value}"
        declared = [c.label for c in case.checks] + [f"judge.{c.name}" for c in case.criteria]
        return EvaluationRecord(
            record_id=record_id, case=case, response=response,
            metrics=tuple(
                MetricOutcome(name=n, kind="judge" if n.startswith("judge.") else "deterministic",
                              outcome=Outcome.NOT_EVALUATED, explanation="no complete model response")
                for n in declared
            ),
            status=RecordStatus.MODEL_ERROR,
            reasons=(f"no complete response from {response.model}: {why}",),
        )

    def run_case(self, case: TestCase) -> EvaluationRecord:
        return self.evaluate_response(case, self.invoke(case))

    # -- suites ------------------------------------------------------------------------------------

    def run(self, cases: Sequence[TestCase] | Dataset, *, dataset_id: str = "adhoc") -> SuiteReport:
        started = utc_now()
        fingerprint = None
        if isinstance(cases, Dataset):
            dataset_id, fingerprint, cases = cases.dataset_id, cases.fingerprint, cases.cases
        records = [self.run_case(case) for case in cases]
        return self.summarize(records, dataset_id=dataset_id, started_at=started, fingerprint=fingerprint)

    def summarize(
        self,
        records: Iterable[EvaluationRecord],
        *,
        dataset_id: str,
        started_at=None,  # noqa: ANN001 - datetime
        fingerprint: str | None = None,
    ) -> SuiteReport:
        """Aggregate records (new or re-evaluated) into a suite report with gates applied."""
        items = tuple(records)
        aggregates = gates.aggregate(items)
        counts = {status.value: sum(r.status is status for r in items) for status in RecordStatus}
        counts["total"] = len(items)
        return SuiteReport(
            run_id=self.run_id,
            evaluation_id=uuid.uuid4().hex[:12],
            started_at=started_at or utc_now(),
            finished_at=utc_now(),
            dataset_id=dataset_id,
            dataset_fingerprint=fingerprint,
            config=self.config,
            records=items,
            aggregates=aggregates,
            counts=counts,
            gates=gates.suite_gates(self.config.suite_thresholds, aggregates),
        )


def _status(metrics: Sequence[MetricOutcome], case_gates) -> tuple[RecordStatus, list[str]]:  # noqa: ANN001
    errors = [f"{m.name}: {m.explanation}" for m in metrics if m.outcome is Outcome.ERROR]
    failures = [f"{m.name}: {m.explanation}" for m in metrics if m.outcome is Outcome.FAILED]
    failures += [f"gate {g.metric}: {g.reason}" for g in case_gates if g.required and g.kind == "threshold_failed"]
    unavailable = [f"gate {g.metric}: {g.reason}" for g in case_gates if g.required and g.kind == "metric_unavailable"]
    if failures:  # a demonstrated failure outranks an evaluator problem; both are listed
        return RecordStatus.FAILED, failures + errors + unavailable
    if errors or unavailable:
        return RecordStatus.EVALUATOR_ERROR, errors + unavailable
    return RecordStatus.PASSED, []
