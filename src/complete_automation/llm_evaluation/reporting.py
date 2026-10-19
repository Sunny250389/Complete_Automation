"""Publishing evaluation results through the shared-core contracts.

No new result format or storage: the suite becomes ``MetricResult`` entries (``llm.<metric>``) and
JSON metadata on the test's ``TestResult`` (through the core ``ResultRecorder``), plus artifacts in
the shared ``ArtifactStore`` under ``llm_evaluation/``:

* ``evaluation-<id>.json``: the full report, with configuration, per-case records and gates;
* ``responses-<id>.jsonl``: the captured model responses, one per line;
* ``cases-<id>.csv`` and ``metrics-<id>.csv``: per-case and aggregate summaries.

Everything is passed through the shared ``Redactor`` before it is stored. ``persist_raw=False``
withholds prompts, responses, references, context and any explanation text that quotes them, and
keeps hashes, lengths, scores, outcomes, latency and token counts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.redaction import Redactor
from complete_automation.llm_evaluation.schemas import (
    EvaluationRecord,
    JudgeResult,
    Message,
    MetricOutcome,
    ModelRequest,
    ModelResponse,
    SuiteReport,
    TestCase,
)

SUBDIR = "llm_evaluation"


def _withheld(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"<withheld sha256:{digest} chars:{len(text)}>"


def _request(request: ModelRequest) -> ModelRequest:
    return request.model_copy(
        update={"messages": tuple(Message(role=m.role, content=_withheld(m.content)) for m in request.messages)}
    )


def _response(response: ModelResponse) -> ModelResponse:
    return response.model_copy(
        update={
            "request": _request(response.request),
            "text": _withheld(response.text) if response.text else "",
            "thinking": _withheld(response.thinking) if response.thinking else None,
        }
    )


def _metric(metric: MetricOutcome) -> MetricOutcome:
    return metric.model_copy(update={"explanation": "", "details": {}})


def _judge(result: JudgeResult) -> JudgeResult:
    return result.model_copy(update={"rationale": "", "evidence": (), "raw_output": None})


def _case(case: TestCase) -> TestCase:
    return case.model_copy(
        update={
            "prompt": _withheld(case.prompt),
            "system_prompt": _withheld(case.system_prompt) if case.system_prompt else None,
            "reference": _withheld(case.reference) if case.reference else None,
            "context": tuple(_withheld(c) for c in case.context),
        }
    )


def _record(record: EvaluationRecord) -> EvaluationRecord:
    return record.model_copy(
        update={
            "case": _case(record.case),
            "response": _response(record.response),
            "metrics": tuple(_metric(m) for m in record.metrics),
            "judge_results": tuple(_judge(j) for j in record.judge_results),
            "reasons": (f"{record.status.value} (details withheld)",) if record.reasons else (),
        }
    )


def withhold_raw(report: SuiteReport) -> SuiteReport:
    """A copy of ``report`` without prompts, responses, references, context or quoting explanations."""
    return report.model_copy(update={"records": tuple(_record(r) for r in report.records)})


def report_to_dict(report: SuiteReport, redactor: Redactor | None = None, *, persist_raw: bool = True) -> dict[str, Any]:
    data = (report if persist_raw else withhold_raw(report)).model_dump(mode="json")
    return redactor.redact(data) if redactor is not None else data


def responses_jsonl(report: SuiteReport, redactor: Redactor | None = None, *, persist_raw: bool = True) -> str:
    lines = []
    for record in report.records:
        response = record.response if persist_raw else _response(record.response)
        item = {"record_id": record.record_id, "case_id": record.case.case_id,
                "response": response.model_dump(mode="json")}
        if redactor is not None:
            item = redactor.redact(item)
        lines.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def metrics_csv(report: SuiteReport) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["metric", "value"])
    for name, value in sorted(report.aggregates.items()):
        writer.writerow([name, "" if value is None else repr(value)])
    return buffer.getvalue()


def cases_csv(report: SuiteReport) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["case_id", "status", "request_status", "latency_seconds", "output_tokens", "failed_or_error_metrics"])
    for r in report.records:
        bad = [m.name for m in r.metrics if m.outcome.value in ("failed", "error", "not_evaluated")]
        writer.writerow([
            r.case.case_id, r.status.value, r.response.status.value, f"{r.response.latency_seconds:.3f}",
            "" if r.response.output_tokens is None else r.response.output_tokens, ";".join(bad),
        ])
    return buffer.getvalue()


def publish_report(
    report: SuiteReport,
    store: ArtifactStore,
    *,
    recorder: ResultRecorder | None = None,
    redactor: Redactor | None = None,
    persist_raw: bool = True,
) -> tuple[SuiteReport, tuple[ArtifactRef, ...]]:
    """Store the artifacts; returns the report (listing the non-JSON artifacts) and all references."""
    meta: dict[str, Any] = {"evaluation_id": report.evaluation_id, "raw_persisted": persist_raw}
    eid = report.evaluation_id
    refs = [
        store.write_text(f"responses-{eid}.jsonl", responses_jsonl(report, redactor, persist_raw=persist_raw),
                         subdir=SUBDIR, media_type="application/x-ndjson", metadata=meta),
        store.write_text(f"cases-{eid}.csv", cases_csv(report), subdir=SUBDIR, media_type="text/csv", metadata=meta),
        store.write_text(f"metrics-{eid}.csv", metrics_csv(report), subdir=SUBDIR, media_type="text/csv", metadata=meta),
    ]
    published = report.model_copy(update={"artifacts": tuple(refs)})
    refs.append(store.write_json(f"evaluation-{eid}.json",
                                 report_to_dict(published, redactor, persist_raw=persist_raw),
                                 subdir=SUBDIR, metadata=meta))
    if recorder is not None:
        for ref in refs:
            recorder.attach(ref)
    return published, tuple(refs)


def record_result(recorder: ResultRecorder, report: SuiteReport) -> None:
    """Add the evaluation to the current test's result: metrics with their gate verdicts, and a summary."""
    gated = {o.metric: o for o in report.gates.outcomes if o.required}
    for name, value in sorted(report.aggregates.items()):
        gate = gated.get(name)
        if value is not None:
            recorder.add_metric(
                f"llm.{name}", value, threshold=gate.threshold if gate else None,
                passed=(gate.kind == "passed") if gate else None,
            )
    for name, gate in gated.items():
        if report.aggregates.get(name) is None:
            recorder.add_metric(f"llm.{name}", "not_computed", passed=False)
    config = report.config
    first = report.records[0].response if report.records else None
    recorder.add_metadata(llm_evaluation={
        "evaluation_id": report.evaluation_id,
        "dataset_id": report.dataset_id,
        "model": {"provider": config.model.provider, "name": config.model.model,
                  "identifier": first.model_identifier if first else None,
                  "endpoint": config.model.base_url},
        "judge": ({"provider": config.judge.model.provider, "name": config.judge.model.model}
                  if config.judge else None),
        "template": config.template.label if config.template else None,
        "counts": report.counts,
        "passed": report.passed,
        "failure_reasons": list(report.failure_reasons)[:20],
        "warnings": list(report.warnings),
    })
