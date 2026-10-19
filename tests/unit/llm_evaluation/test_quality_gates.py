"""Thresholds, aggregates, failure classification and baseline comparison."""

import json

import httpx
import pytest

from complete_automation.llm_evaluation import (
    BaselineError,
    BaselineRule,
    CheckSpec,
    CriterionSpec,
    EvaluationRunner,
    EvaluatorError,
    JudgeEvaluator,
    ModelInvocationError,
    QualityGateError,
    RecordStatus,
    TestCase,
    ThresholdRule,
    apply_baseline,
)
from complete_automation.llm_evaluation.baseline import Baseline
from complete_automation.llm_evaluation.gates import compare_to_baseline, evaluate_rules
from tests.unit.llm_evaluation.helpers import JUDGE, Script, client_for, config, judge_reply, reply

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]


def rule(metric: str, op: str, value: float, **kw) -> ThresholdRule:
    return ThresholdRule(metric=metric, operator=op, value=value, **kw)


@pytest.mark.parametrize(
    ("op", "observed", "expected"),
    [("ge", 0.9, "passed"), ("ge", 0.89, "threshold_failed"), ("gt", 0.9, "threshold_failed"), ("gt", 0.91, "passed"),
     ("le", 0.9, "passed"), ("le", 0.91, "threshold_failed"), ("lt", 0.9, "threshold_failed"), ("lt", 0.89, "passed"),
     ("eq", 0.9, "passed"), ("eq", 0.8, "threshold_failed")],
)
def test_operators(op: str, observed: float, expected: str) -> None:
    (outcome,) = evaluate_rules([rule("m", op, 0.9)], {"m": observed}, "suite")
    assert outcome.kind == expected


def test_eq_uses_tolerance() -> None:
    assert evaluate_rules([rule("m", "eq", 1.0, tolerance=0.05)], {"m": 1.04}, "suite")[0].kind == "passed"


def test_missing_metric_never_passes() -> None:
    (outcome,) = evaluate_rules([rule("never_measured", "ge", 0.0)], {}, "suite")
    assert outcome.kind == "metric_unavailable" and "cannot be confirmed" in outcome.reason


def make_runner(texts, *, judge_replies=(), **cfg):
    target = Script(tags=("target-model",))
    target.queue.extend(texts)
    judge = Script()
    judge.queue.extend(judge_replies)
    kwargs = {"judge": JudgeEvaluator(client_for(judge), JUDGE)} if judge_replies else {}
    if judge_replies:
        cfg["judge"] = JUDGE
    return EvaluationRunner(client_for(target), config(**cfg), **kwargs)


def json_case(case_id: str) -> TestCase:
    return TestCase(case_id=case_id, prompt="give json", checks=(CheckSpec(type="json_valid"),))


def test_suite_aggregates_and_pass_rate_gate() -> None:
    cfg = dict(suite_thresholds=(rule("json_valid.pass_rate", "ge", 1.0),))
    report = make_runner([reply('{"a":1}'), reply("oops")], **cfg).run([json_case("a"), json_case("b")])
    assert report.aggregates["json_valid.pass_rate"] == 0.5 and report.aggregates["case_pass_rate"] == 0.5
    assert not report.gates.passed and report.gates.outcomes[0].kind == "threshold_failed"
    assert report.counts == {"passed": 1, "failed": 1, "model_error": 0, "evaluator_error": 0, "total": 2}
    assert not report.passed
    with pytest.raises(QualityGateError, match="b \\[failed\\]") as info:
        report.assert_passed()
    assert "json_valid.pass_rate" in str(info.value)


def test_passing_suite_passes() -> None:
    cfg = dict(suite_thresholds=(rule("json_valid.pass_rate", "ge", 1.0), rule("latency_seconds.max", "le", 60)))
    report = make_runner([reply('{"a":1}')], **cfg).run([json_case("a")])
    assert report.passed and report.gates.passed
    report.assert_passed()


def test_suite_rule_on_a_metric_nobody_measured_fails_the_suite() -> None:
    report = make_runner([reply('{"a":1}')], suite_thresholds=(rule("judge.correctness.mean", "ge", 4),)).run([json_case("a")])
    assert not report.passed and report.gates.outcomes[0].kind == "metric_unavailable"


def test_optional_rule_does_not_fail_the_suite_when_unavailable() -> None:
    report = make_runner([reply('{"a":1}')], suite_thresholds=(rule("nope", "ge", 1, required=False),)).run([json_case("a")])
    assert report.passed


def test_empty_evaluation_is_not_a_pass() -> None:
    report = make_runner([]).run([])
    assert not report.passed and "no test cases" in report.failure_reasons[0]


def test_judge_errors_are_not_counted_as_successful_scores() -> None:
    case = lambda cid: TestCase(case_id=cid, prompt="q", reference="r", criteria=(CriterionSpec(name="correctness", min_score=3),))  # noqa: E731
    runner = make_runner([reply("a"), reply("b")], judge_replies=[judge_reply("correctness", 5), reply("garbage")],
                         suite_thresholds=(rule("judge.correctness.pass_rate", "ge", 1.0), rule("judge.correctness.errors", "eq", 0)))
    report = runner.run([case("ok"), case("bad")])
    assert report.aggregates["judge.correctness.pass_rate"] == 0.5  # the error counts against, not for
    assert report.aggregates["judge.correctness.errors"] == 1.0
    assert report.aggregates["judge.correctness.mean"] == 5.0  # error contributed no score
    assert report.counts["evaluator_error"] == 1 and not report.passed


def test_model_failures_evaluator_failures_and_threshold_failures_raise_different_errors() -> None:
    model_down = make_runner([httpx.ConnectError("x")]).run([json_case("a")])
    with pytest.raises(ModelInvocationError):
        model_down.assert_passed()
    assert model_down.aggregates["model_error_rate"] == 1.0

    case = TestCase(case_id="e", prompt="q", reference="r", criteria=(CriterionSpec(name="correctness", min_score=3),))
    evaluator_down = make_runner([reply("a")], judge_replies=[reply("garbage")]).run([case])
    with pytest.raises(EvaluatorError):
        evaluator_down.assert_passed()

    bad = make_runner([reply("nope")]).run([json_case("t")])
    with pytest.raises(QualityGateError):
        bad.assert_passed()
    assert not isinstance(QualityGateError("x"), (ModelInvocationError, EvaluatorError))


def test_model_error_cases_count_against_pass_rates() -> None:
    report = make_runner([reply('{"a":1}'), httpx.ConnectError("x")]).run([json_case("a"), json_case("b")])
    assert report.aggregates["json_valid.pass_rate"] == 0.5
    assert [r.status for r in report.records] == [RecordStatus.PASSED, RecordStatus.MODEL_ERROR]


def test_case_level_threshold_on_latency_and_unmeasured_tokens() -> None:
    case = TestCase(case_id="t", prompt="q", checks=(CheckSpec(type="not_empty"),),
                    thresholds=(rule("latency_seconds", "le", 1000), rule("output_tokens", "le", 100)))
    no_tokens = httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}, "done": True})
    record = make_runner([no_tokens]).run_case(case)
    assert record.status is RecordStatus.EVALUATOR_ERROR  # token count unavailable: not inferred, not passed
    assert {g.metric: g.kind for g in record.gates} == {"latency_seconds": "passed", "output_tokens": "metric_unavailable"}
    with_tokens = make_runner([reply("x", eval_count=40)]).run_case(case)
    assert with_tokens.status is RecordStatus.PASSED
    over = make_runner([reply("x", eval_count=400)]).run_case(case)
    assert over.status is RecordStatus.FAILED and "output_tokens" in over.reasons[0]


def test_demonstrated_failure_outranks_an_evaluator_problem_and_lists_both() -> None:
    case = TestCase(case_id="m", prompt="q", checks=(CheckSpec(type="contains", value="x"), CheckSpec(type="regex", value="(")))
    record = make_runner([reply("y")]).run_case(case)
    assert record.status is RecordStatus.FAILED and len(record.reasons) == 2


# ---- baseline


def test_baseline_reports_changes_but_only_rules_fail() -> None:
    result = compare_to_baseline("b", {"acc": 0.9, "lat": 2.0}, {"acc": 0.8, "lat": 3.0})
    assert result.passed and {c.outcome for c in result.comparisons} == {"changed"}


def test_baseline_rule_allows_bounded_drop_and_flags_regression() -> None:
    rules = [BaselineRule(metric="acc", max_drop=0.05), BaselineRule(metric="lat", direction="lower_is_better", max_drop=0.5)]
    ok = compare_to_baseline("b", {"acc": 0.9, "lat": 2.0}, {"acc": 0.86, "lat": 2.4}, rules)
    assert ok.passed
    bad = compare_to_baseline("b", {"acc": 0.9, "lat": 2.0}, {"acc": 0.8, "lat": 3.0}, rules)
    assert not bad.passed and {c.metric for c in bad.comparisons if c.outcome == "regressed"} == {"acc", "lat"}
    better = compare_to_baseline("b", {"acc": 0.9}, {"acc": 1.0}, [BaselineRule(metric="acc")])
    assert better.passed


def test_baseline_rule_on_missing_value_is_unavailable_and_fails() -> None:
    result = compare_to_baseline("b", {"acc": 0.9}, {}, [BaselineRule(metric="acc")])
    assert not result.passed and result.comparisons[0].outcome == "unavailable"
    with pytest.raises(BaselineError):
        compare_to_baseline("b", {}, {}, [BaselineRule(metric="ghost")])


def test_baseline_roundtrip_and_application_to_a_report(tmp_path) -> None:
    cfg = dict(baseline_rules=(BaselineRule(metric="json_valid.pass_rate"),))
    first = make_runner([reply('{"a":1}')], **cfg).run([json_case("a")], dataset_id="d")
    baseline = Baseline.from_report(first, "base-1")
    path = tmp_path / "base.json"
    path.write_text(json.dumps(baseline.model_dump(mode="json")), encoding="utf-8")
    loaded = Baseline.load(tmp_path, "base.json")
    assert loaded == baseline

    worse = make_runner([reply("oops")], **cfg).run([json_case("a")], dataset_id="d")
    compared = apply_baseline(worse, loaded)
    assert compared.baseline and not compared.baseline.passed and not compared.passed
    assert any("json_valid.pass_rate" in r for r in compared.failure_reasons)

    same = apply_baseline(make_runner([reply('{"b":2}')], **cfg).run([json_case("a")], dataset_id="d"), loaded)
    assert same.passed  # different text, same quality: a changed output alone is not a failure

    with pytest.raises(BaselineError, match="dataset"):
        apply_baseline(make_runner([reply("{}")]).run([json_case("a")], dataset_id="other"), loaded)


def test_invalid_baseline_file_is_a_clear_error(tmp_path) -> None:
    (tmp_path / "bad.json").write_text('{"nope": 1}', encoding="utf-8")
    with pytest.raises(BaselineError, match="invalid baseline"):
        Baseline.load(tmp_path, "bad.json")
    with pytest.raises(BaselineError):
        Baseline.load(tmp_path, "missing.json")
