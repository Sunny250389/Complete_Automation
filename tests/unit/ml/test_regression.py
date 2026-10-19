"""Regression metrics against hand-computed values."""

import math

import pytest

from tests.unit.ml.helpers import evaluate_arrays

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]

Y = [3.0, -0.5, 2.0, 7.0]
P = [2.5, 0.0, 2.0, 8.0]  # absolute errors 0.5, 0.5, 0.0, 1.0


def run(y=Y, p=P, **config):
    return evaluate_arrays("regression", y, p, **config)


def test_error_metrics() -> None:
    report = run()
    assert report.value("mae") == pytest.approx(0.5)
    assert report.value("mse") == pytest.approx(0.375)  # (0.25 + 0.25 + 0 + 1) / 4
    assert report.value("rmse") == pytest.approx(math.sqrt(0.375))
    assert report.value("median_ae") == pytest.approx(0.5)


def test_r2_and_mape() -> None:
    report = run()
    # mean(y) = 2.875; SS_tot = 29.1875; SS_res = 1.5
    assert report.value("r2") == pytest.approx(1 - 1.5 / 29.1875)
    assert report.value("mape") == pytest.approx((0.5 / 3 + 1.0 + 0.0 + 1 / 7) / 4)


def test_perfect_predictions() -> None:
    report = run(p=Y)
    assert report.value("rmse") == 0.0 and report.value("r2") == 1.0


def test_mape_with_a_zero_target_is_undefined_by_default() -> None:
    report = run(y=[0.0, 2.0, 4.0], p=[1.0, 2.0, 3.0])
    mape = report.metric("mape")
    assert mape.status.value == "undefined" and mape.value is None
    assert "1 of 3" in mape.reason
    assert report.value("mae") == pytest.approx(2 / 3)  # other metrics are unaffected


def test_mape_can_exclude_zero_targets_and_says_so() -> None:
    report = run(y=[0.0, 2.0, 4.0], p=[1.0, 2.0, 3.0], mape_zero_policy="exclude")
    mape = report.metric("mape")
    assert mape.value == pytest.approx((0.0 + 0.25) / 2)
    assert mape.detail == {"excluded_zero_targets": 1, "samples_used": 2}


def test_mape_treats_near_zero_targets_with_the_epsilon() -> None:
    report = run(y=[1e-9, 2.0], p=[1.0, 2.0], mape_epsilon=1e-6)
    assert report.metric("mape").status.value == "undefined"


def test_r2_is_undefined_for_constant_targets() -> None:
    report = run(y=[2.0, 2.0, 2.0], p=[1.0, 2.0, 3.0])
    assert report.metric("r2").status.value == "undefined"
    assert "constant" in report.metric("r2").reason


def test_r2_can_be_negative() -> None:
    assert run(y=[1.0, 2.0, 3.0], p=[3.0, 2.0, 1.0]).value("r2") == pytest.approx(-3.0)  # 1 - SS_res (8) / SS_tot (2)


def test_regression_metrics_declare_their_direction() -> None:
    report = run()
    assert report.metric("rmse").direction.value == "lower_is_better"
    assert report.metric("r2").direction.value == "higher_is_better"


def test_non_finite_predictions_block_the_evaluation() -> None:
    report = run(p=[2.5, float("inf"), 2.0, 8.0])
    assert not report.passed and report.metrics == ()
    assert [f.code for f in report.errors] == ["non_finite_predictions"]


def test_missing_predictions_block_the_evaluation() -> None:
    report = run(p=[2.5, None, 2.0, 8.0])
    assert [f.code for f in report.errors] == ["missing_predictions"]
