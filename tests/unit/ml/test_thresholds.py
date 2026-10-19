"""Quality gates and baseline comparison."""

import pytest
from pydantic import ValidationError

from complete_automation.ml import (
    Baseline,
    BaselineError,
    BaselineRule,
    EvaluationConfig,
    Evaluator,
    MetricUnavailableError,
    MLConfigurationError,
    QualityGateError,
    TaskType,
    ThresholdRule,
    ValidationFailedError,
    compare_to_baseline,
    evaluate_gates,
    parse_thresholds,
)
from complete_automation.ml.schemas import MetricStatus, MetricValue
from tests.unit.ml.helpers import evaluate_arrays

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]


def value(name, number):
    return MetricValue(name=name, status=MetricStatus.COMPUTED, value=number)


def gate(rule, observed):
    return evaluate_gates([rule], [value(rule.metric, observed)], TaskType.CLASSIFICATION).outcomes[0]


@pytest.mark.parametrize(
    ("rule", "observed", "expected"),
    [
        (ThresholdRule.at_least("m", 0.9), 0.9, True),
        (ThresholdRule.at_least("m", 0.9), 0.8999, False),
        (ThresholdRule.above("m", 0.9), 0.9, False),
        (ThresholdRule.above("m", 0.9), 0.9001, True),
        (ThresholdRule.at_most("m", 5.0), 5.0, True),
        (ThresholdRule.at_most("m", 5.0), 5.01, False),
        (ThresholdRule.below("m", 5.0), 5.0, False),
        (ThresholdRule.below("m", 5.0), 4.99, True),
        (ThresholdRule.equals("m", 1.0), 1.0, True),
        (ThresholdRule.equals("m", 1.0), 1.0001, False),
        (ThresholdRule.equals("m", 1.0, tolerance=0.01), 1.005, True),
    ],
)
def test_every_operator_at_and_around_its_boundary(rule, observed, expected) -> None:
    assert gate(rule, observed).passed is expected


def test_a_failure_names_metric_observed_operator_threshold_and_reason() -> None:
    outcome = gate(ThresholdRule.at_least("f1", 0.9), 0.85)
    assert (outcome.metric, outcome.observed, outcome.operator.value, outcome.threshold) == ("f1", 0.85, "ge", 0.9)
    assert outcome.kind == "threshold_failed"
    assert "observed 0.85" in outcome.reason and ">= 0.9" in outcome.reason


def test_a_missing_metric_fails_instead_of_passing() -> None:
    summary = evaluate_gates([ThresholdRule.at_least("nope", 0.1)], [], TaskType.CLASSIFICATION)
    assert not summary.passed
    assert summary.outcomes[0].kind == "metric_unavailable" and summary.unavailable


def test_an_undefined_metric_is_a_calculation_failure_not_a_threshold_failure() -> None:
    report = evaluate_arrays("regression", [0.0, 2.0], [1.0, 2.0], metrics=("mape",),
                             thresholds=(ThresholdRule.at_most("mape", 0.5),))
    outcome = report.gates.outcomes[0]
    assert outcome.kind == "metric_unavailable" and outcome.observed is None
    assert "undefined" in outcome.reason and not report.gates.threshold_failures
    with pytest.raises(MetricUnavailableError):
        report.assert_passed()


def test_a_threshold_failure_raises_the_quality_gate_error() -> None:
    report = evaluate_arrays("regression", [1.0, 2.0], [2.0, 4.0], metrics=("mae",),
                             thresholds=(ThresholdRule.at_most("mae", 1.0),))
    assert not report.passed
    with pytest.raises(QualityGateError, match=r"mae .*observed 1\.5"):
        report.assert_passed()


def test_aggregate_requires_all_required_rules() -> None:
    rules = (ThresholdRule.at_most("mae", 5.0), ThresholdRule.at_most("rmse", 0.001))
    report = evaluate_arrays("regression", [1.0, 2.0], [2.0, 4.0], metrics=("mae", "rmse"), thresholds=rules)
    assert [o.passed for o in report.gates.outcomes] == [True, False] and not report.gates.passed


def test_an_optional_rule_is_reported_but_does_not_fail_the_gate() -> None:
    rules = (ThresholdRule.at_most("mae", 0.0, required=False),)
    report = evaluate_arrays("regression", [1.0], [2.0], metrics=("mae",), thresholds=rules)
    assert report.gates.outcomes[0].kind == "threshold_failed" and report.gates.passed and report.passed


def test_a_rule_against_the_metric_direction_is_flagged() -> None:
    summary = evaluate_gates([ThresholdRule.at_least("rmse", 5.0)], [value("rmse", 6.0)], TaskType.REGRESSION)
    assert summary.passed and "lower is better" in summary.warnings[0]


def test_no_rules_means_no_gate() -> None:
    assert evaluate_arrays("regression", [1.0], [1.0], metrics=("mae",)).gates is None


def test_threshold_configuration_is_validated() -> None:
    assert parse_thresholds({"f1": {"ge": 0.9}, "rmse": {"le": 5, "gt": 0}})[2].operator.value == "gt"
    with pytest.raises(MLConfigurationError):
        parse_thresholds({"f1": {"approximately": 0.9}})
    with pytest.raises(MLConfigurationError):
        parse_thresholds({"f1": 0.9})
    with pytest.raises(ValidationError):
        ThresholdRule(metric="f1", operator="ge", value=float("nan"))
    with pytest.raises(ValidationError):
        ThresholdRule(metric="f1", operator="ge", value=0.5, tolerance=0.1)  # tolerance is for eq only


def test_unknown_metrics_and_unsupported_tasks_are_configuration_errors() -> None:
    with pytest.raises(MLConfigurationError, match="rmse"):
        Evaluator(EvaluationConfig(task="classification", metrics=("rmse",)))
    with pytest.raises(ValidationError):
        EvaluationConfig(task="translation")
    with pytest.raises(ValidationError):
        EvaluationConfig(task="classification", averaging="binary")  # no positive_label


def test_blocking_findings_raise_the_validation_error() -> None:
    report = evaluate_arrays("regression", [1.0, 2.0], [1.0, None])
    with pytest.raises(ValidationFailedError):
        report.assert_passed()


# --- baselines --------------------------------------------------------------------------------


def baseline(**metrics):
    return Baseline(baseline_id="b1", task=TaskType.CLASSIFICATION, model_id="m", dataset_id="d",
                    dataset_fingerprint="x", metrics=metrics)


def compare(base, current, rules=()):
    items = [
        value(k, v) if v is not None else MetricValue(name=k, status=MetricStatus.UNDEFINED, reason="r")
        for k, v in current.items()
    ]
    return compare_to_baseline(base, items, rules, task=TaskType.CLASSIFICATION)


def entry(result, name):
    return next(c for c in result.comparisons if c.metric == name)


def test_differences_are_reported_without_failing() -> None:
    result = compare(baseline(accuracy=0.8, log_loss=0.5), {"accuracy": 0.6, "log_loss": 0.7})
    accuracy = entry(result, "accuracy")
    assert accuracy.absolute_difference == pytest.approx(-0.2) and accuracy.relative_difference == pytest.approx(-0.25)
    assert accuracy.outcome == "changed" and result.passed  # a change alone is not a regression


def test_rules_use_each_metric_direction() -> None:
    rules = [BaselineRule(metric="accuracy", max_drop=0.05), BaselineRule(metric="log_loss", max_drop=0.05)]
    result = compare(baseline(accuracy=0.8, log_loss=0.5), {"accuracy": 0.79, "log_loss": 0.6}, rules)
    assert entry(result, "accuracy").outcome == "passed"     # dropped 0.01, allowed 0.05
    assert entry(result, "log_loss").outcome == "regressed"  # rose 0.1: worse, because lower is better
    assert not result.passed


def test_improvements_are_never_regressions() -> None:
    result = compare(baseline(accuracy=0.8), {"accuracy": 0.95}, [BaselineRule(metric="accuracy", max_drop=0.0)])
    assert entry(result, "accuracy").outcome == "passed" and result.passed


def test_relative_rules_and_a_zero_baseline() -> None:
    rule = BaselineRule(metric="accuracy", max_relative_drop=0.10)
    assert compare(baseline(accuracy=0.8), {"accuracy": 0.73}, [rule]).passed is True   # 8.75 % drop
    assert compare(baseline(accuracy=0.8), {"accuracy": 0.70}, [rule]).passed is False  # 12.5 % drop
    zero = compare(baseline(accuracy=0.0), {"accuracy": 0.0}, [rule])
    assert entry(zero, "accuracy").relative_difference is None
    assert entry(zero, "accuracy").outcome == "unavailable" and not zero.passed


def test_an_undefined_baseline_or_current_metric_cannot_pass_a_rule() -> None:
    rule = [BaselineRule(metric="accuracy", max_drop=0.1)]
    undefined_current = compare(baseline(accuracy=0.8), {"accuracy": None}, rule)
    undefined_baseline = compare(baseline(accuracy=None), {"accuracy": 0.9}, rule)
    for result in (undefined_current, undefined_baseline):
        assert entry(result, "accuracy").outcome == "unavailable" and not result.passed
    assert compare(baseline(accuracy=None), {"accuracy": 0.9}).passed  # without a rule it is just a change


def test_baseline_mistakes_are_reported() -> None:
    with pytest.raises(BaselineError, match="neither"):
        compare(baseline(accuracy=0.8), {"accuracy": 0.8}, [BaselineRule(metric="ghost", max_drop=0.1)])
    with pytest.raises(BaselineError, match="regression"):
        compare_to_baseline(baseline(accuracy=0.8), [], task=TaskType.REGRESSION)
    with pytest.raises(ValidationError):
        BaselineRule(metric="accuracy")
    with pytest.raises(BaselineError):
        Baseline.loads("{not json")
