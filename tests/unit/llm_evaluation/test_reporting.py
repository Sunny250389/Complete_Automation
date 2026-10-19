"""Report serialisation, artifacts, redaction, raw-data withholding, and shared-core integration."""

import json

import pytest

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.redaction import Redactor
from complete_automation.llm_evaluation import (
    CheckSpec,
    CriterionSpec,
    EvaluationRunner,
    JudgeEvaluator,
    SuiteReport,
    TestCase,
    ThresholdRule,
    publish_report,
    record_result,
)
from complete_automation.llm_evaluation.reporting import cases_csv, metrics_csv, report_to_dict, withhold_raw
from tests.unit.llm_evaluation.helpers import JUDGE, Script, client_for, config, judge_reply, reply

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]

PROMPT = "Summarise the ticket for customer Alice"
ANSWER = "Ticket summary. config: password=hunter2 and api_key=sk-live-12345"


@pytest.fixture
def report() -> SuiteReport:
    target = Script(tags=("target-model",))
    target.queue.append(reply(ANSWER, eval_count=9))
    judge = Script()
    judge.queue.append(judge_reply("correctness", 5, rationale="quotes: " + ANSWER, evidence=[ANSWER]))
    case = TestCase(
        case_id="t1", prompt=PROMPT, reference="REFERENCE-TEXT", context=("CONTEXT-TEXT",),
        checks=(CheckSpec(type="contains", value="summary", case_sensitive=False),),
        criteria=(CriterionSpec(name="correctness", min_score=3),),
    )
    cfg = config(judge=JUDGE, suite_thresholds=(ThresholdRule(metric="case_pass_rate", operator="ge", value=1.0),))
    return EvaluationRunner(client_for(target), cfg, judge=JudgeEvaluator(client_for(judge), JUDGE), run_id="run-1").run([case], dataset_id="ds")


def test_report_contains_everything_a_reader_needs(report: SuiteReport) -> None:
    data = report_to_dict(report)
    assert (data["run_id"], data["dataset_id"], data["schema_version"]) == ("run-1", "ds", "1.0")
    record = data["records"][0]
    assert record["case"]["reference"] == "REFERENCE-TEXT" and record["case"]["context"] == ["CONTEXT-TEXT"]
    assert record["response"]["text"] == ANSWER and record["response"]["output_tokens"] == 9
    assert record["response"]["model"] == "target-model" and data["config"]["judge"]["model"]["model"] == "judge-model"
    assert record["response"]["request"]["config"]["generation"]["temperature"] == 0.0
    assert record["status"] == "passed" and data["gates"]["passed"] is True
    assert SuiteReport.model_validate(data) == report  # round-trips


def test_publish_stores_artifacts_through_the_shared_store_and_references_them(report: SuiteReport, tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    recorder = ResultRecorder()
    published, refs = publish_report(report, store, recorder=recorder)
    names = sorted(r.name for r in refs)
    assert names == sorted([f"{prefix}-{report.evaluation_id}.{ext}" for prefix, ext in
                            [("responses", "jsonl"), ("cases", "csv"), ("metrics", "csv"), ("evaluation", "json")]])
    assert recorder.artifacts == list(refs) and len(published.artifacts) == 3
    for ref in refs:
        assert ref.path.startswith("llm_evaluation/") and store.resolve(ref).is_file() and len(ref.sha256) == 64
    stored = json.loads(store.resolve(next(r for r in refs if r.name.startswith("evaluation"))).read_text(encoding="utf-8"))
    assert len(stored["artifacts"]) == 3
    lines = store.resolve(next(r for r in refs if r.name.startswith("responses"))).read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["case_id"] == "t1"


def test_secrets_are_redacted_in_every_stored_artifact(report: SuiteReport, tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    _, refs = publish_report(report, store, redactor=Redactor())
    for ref in refs:
        text = store.resolve(ref).read_text(encoding="utf-8")
        assert "hunter2" not in text and "sk-live-12345" not in text, ref.name
    assert "password=***" in store.resolve(refs[0]).read_text(encoding="utf-8")


def test_withholding_raw_data_keeps_metrics_and_metadata_but_no_content(report: SuiteReport, tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    _, refs = publish_report(report, store, redactor=Redactor(), persist_raw=False)
    for ref in refs:
        text = store.resolve(ref).read_text(encoding="utf-8")
        for secret in (PROMPT, ANSWER, "REFERENCE-TEXT", "CONTEXT-TEXT", "Ticket summary"):
            assert secret not in text, (ref.name, secret)
    evaluation = json.loads(store.resolve(next(r for r in refs if r.name.startswith("evaluation"))).read_text(encoding="utf-8"))
    record = evaluation["records"][0]
    assert record["response"]["text"].startswith("<withheld sha256:") and record["response"]["output_tokens"] == 9
    assert record["status"] == "passed" and record["judge_results"][0]["score"] == 5.0
    assert record["judge_results"][0]["rationale"] == "" and record["judge_results"][0]["raw_output"] is None
    assert record["response"]["request"]["messages"][0]["content"].startswith("<withheld")
    assert withhold_raw(report).records[0].response.latency_seconds == report.records[0].response.latency_seconds


def test_csv_summaries(report: SuiteReport) -> None:
    assert "case_pass_rate,1.0" in metrics_csv(report)
    lines = cases_csv(report).splitlines()
    assert lines[0].startswith("case_id,status") and lines[1].startswith("t1,passed,completed")


def test_result_recorder_receives_metrics_with_gate_verdicts_and_metadata(report: SuiteReport) -> None:
    recorder = ResultRecorder()
    record_result(recorder, report)
    metrics = {m.name: m for m in recorder.metrics}
    gate = metrics["llm.case_pass_rate"]
    assert (gate.value, gate.threshold, gate.passed) == (1.0, 1.0, True)
    assert metrics["llm.contains.pass_rate"].passed is None  # not gated: no verdict invented
    meta = recorder.metadata["llm_evaluation"]
    assert meta["model"]["name"] == "target-model" and meta["model"]["identifier"] == "digest-target-model"
    assert meta["judge"]["name"] == "judge-model" and meta["passed"] is True and meta["counts"]["passed"] == 1


def test_unmeasured_gated_metric_is_recorded_as_not_computed() -> None:
    target = Script(tags=("target-model",))
    target.queue.append(reply("x"))
    cfg = config(suite_thresholds=(ThresholdRule(metric="ghost.mean", operator="ge", value=1),))
    case = TestCase(case_id="c", prompt="q", checks=(CheckSpec(type="not_empty"),))
    report = EvaluationRunner(client_for(target), cfg).run([case])
    recorder = ResultRecorder()
    record_result(recorder, report)
    ghost = next(m for m in recorder.metrics if m.name == "llm.ghost.mean")
    assert ghost.value == "not_computed" and ghost.passed is False
