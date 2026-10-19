"""Model adapter contracts, robustness, repeatability, subgroups and custom checks."""

import numpy as np
import pandas as pd
import pytest

from complete_automation.ml import (
    CheckContext,
    Dataset,
    EstimatorAdapter,
    EvaluationConfig,
    Evaluator,
    FunctionAdapter,
    MLConfigurationError,
    ModelAdapter,
    ModelInvocationError,
    PerturbationSpec,
    PrecomputedAdapter,
    RepeatabilityConfig,
    SubgroupConfig,
)
from complete_automation.ml.schemas import CheckResult

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]


def regression_data(n=40):
    x = np.linspace(0.0, 10.0, n)
    frame = pd.DataFrame({"x": x, "group": ["a", "b"] * (n // 2), "y": 2.0 * x})
    return Dataset.from_frame(frame, "reg", features=["x"], target="y")


def classification_data(n=40):
    x = np.linspace(-5.0, 5.0, n)
    frame = pd.DataFrame({"x": x, "group": ["a"] * (n - 4) + ["b"] * 4, "label": (x > 0).astype(int)})
    return Dataset.from_frame(frame, "cls", features=["x"], target="label")


class Doubler:
    """A fitted-looking regressor that records how it was called."""

    def __init__(self):
        self.calls = []

    def predict(self, X):
        self.calls.append(("predict", X.copy()))
        X.iloc[:, 0] = -999.0  # a careless model mutating its input
        return np.asarray(X.index, dtype=float) * 0 + 1.0


class Threshold:
    def __init__(self, cut=0.0):
        self.cut = cut
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        return (X["x"].to_numpy() > self.cut).astype(int)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X["x"].to_numpy()))
        return np.column_stack([1 - p, p])


# --- adapters ---------------------------------------------------------------------------------


def test_the_evaluator_does_not_let_a_model_modify_the_dataset() -> None:
    data = regression_data()
    before = data.frame.copy()
    Evaluator(EvaluationConfig(task="regression")).evaluate(data, EstimatorAdapter(Doubler(), model_id="d"))
    pd.testing.assert_frame_equal(data.frame, before)


def test_estimator_adapter_uses_only_what_the_estimator_offers() -> None:
    adapter = EstimatorAdapter(Doubler(), model_id="d")
    assert not adapter.supports_probabilities and not adapter.supports_scores
    full = EstimatorAdapter(Threshold(), model_id="t")
    assert full.supports_probabilities and not full.supports_scores and full.classes == (0, 1)
    assert full.info.class_name == "Threshold" and full.info.kind == "estimator"


def test_an_object_without_predict_is_rejected() -> None:
    with pytest.raises(MLConfigurationError, match="predict"):
        EstimatorAdapter(object(), model_id="x")


def test_a_failing_model_raises_a_model_invocation_error() -> None:
    def broken(X):
        raise RuntimeError("model server down")

    with pytest.raises(ModelInvocationError, match="model server down"):
        Evaluator(EvaluationConfig(task="regression")).evaluate(
            regression_data(), FunctionAdapter(broken, model_id="broken"))


def test_function_adapter_and_custom_adapter_extension_point() -> None:
    class Constant(ModelAdapter):
        info = FunctionAdapter(lambda X: X, model_id="c").info

        def predict(self, features):
            return np.full(len(features), 1.0)

    report = Evaluator(EvaluationConfig(task="regression", metrics=("mae",))).evaluate(regression_data(), Constant())
    assert report.value("mae") == pytest.approx(np.mean(np.abs(2.0 * np.linspace(0, 10, 40) - 1.0)))


def test_precomputed_probabilities_need_their_class_order() -> None:
    with pytest.raises(MLConfigurationError, match="class order"):
        PrecomputedAdapter([0, 1], probabilities=[[0.5, 0.5], [0.5, 0.5]])


def test_precomputed_adapter_from_frame_columns() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "label": ["n", "p", "n", "p"],
                          "pred": ["n", "p", "p", "p"], "p_n": [0.8, 0.3, 0.4, 0.1], "p_p": [0.2, 0.7, 0.6, 0.9]})
    data = Dataset.from_frame(frame, "pre", features=["x"], target="label")
    adapter = PrecomputedAdapter.from_frame(frame, "pred", probability_columns={"n": "p_n", "p": "p_p"})
    report = Evaluator(EvaluationConfig(task="classification", positive_label="p",
                                        metrics=("accuracy", "roc_auc"))).evaluate(data, adapter)
    assert report.value("accuracy") == 0.75 and report.value("roc_auc") == 1.0
    with pytest.raises(MLConfigurationError, match="nope"):
        PrecomputedAdapter.from_frame(frame, "nope")


# --- robustness -------------------------------------------------------------------------------


def test_perturbations_are_seeded_and_do_not_touch_the_original() -> None:
    from complete_automation.ml.robustness import perturb

    data = regression_data()
    spec = PerturbationSpec(name="noise", kind="gaussian_noise", features=("x",), magnitude=0.05, seed=3)
    first, changed = perturb(data.X, spec)
    second, _ = perturb(data.X, spec)
    pd.testing.assert_frame_equal(first, second)           # same seed, same perturbation
    assert changed == 40 and not first["x"].equals(data.X["x"])
    assert data.frame["x"].equals(pd.Series(np.linspace(0, 10, 40), name="x"))  # original untouched


def test_small_noise_degrades_a_threshold_model_little_and_big_noise_more() -> None:
    data = classification_data(200)
    adapter = EstimatorAdapter(Threshold(), model_id="t")
    mild = PerturbationSpec(name="mild", kind="gaussian_noise", features=("x",), magnitude=0.01, seed=1,
                            gate_metric="accuracy", max_metric_drop=0.05)
    heavy = PerturbationSpec(name="heavy", kind="gaussian_noise", features=("x",), magnitude=1.0, seed=1,
                             gate_metric="accuracy", max_metric_drop=0.01)
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        data, adapter, robustness=[mild, heavy])
    mild_out, heavy_out = report.robustness
    assert mild_out.passed is True and heavy_out.passed is False
    assert mild_out.metric_changes["accuracy"]["original"] == 1.0
    assert heavy_out.metric_changes["accuracy"]["degradation"] > mild_out.metric_changes["accuracy"]["degradation"]
    assert heavy_out.prediction_change > mild_out.prediction_change
    assert not report.passed and any("robustness heavy" in r for r in report.failure_reasons)
    assert mild_out.features == ("x",) and mild_out.seed == 1


def test_robustness_without_limits_reports_but_does_not_judge() -> None:
    spec = PerturbationSpec(name="scale", kind="relative_scale", features=("x",), magnitude=0.5)
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        classification_data(), EstimatorAdapter(Threshold(), model_id="t"), robustness=[spec])
    assert report.robustness[0].passed is None and report.passed


def test_categorical_substitution_and_missing_values() -> None:
    frame = pd.DataFrame({"c": ["x", "y", "x", "z"], "n": [1.0, 2.0, 3.0, 4.0], "t": [0, 1, 0, 1]})
    data = Dataset.from_frame(frame, "cat", features=["c", "n"], target="t")
    seen = []

    def predict(X):
        seen.append(X.copy())
        return np.zeros(len(X), dtype=int)

    adapter = FunctionAdapter(predict, model_id="f")
    substitute = PerturbationSpec(name="sub", kind="substitute_category", features=("c",), mapping={"x": "z"})
    missing = PerturbationSpec(name="gap", kind="set_missing", features=("n",), magnitude=1.0, allow_missing_inputs=True)
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        data, adapter, robustness=[substitute, missing])
    assert list(seen[1]["c"]) == ["z", "y", "z", "z"] and report.robustness[0].cells_changed == 2
    assert seen[2]["n"].isna().all()


def test_invalid_perturbation_configuration() -> None:
    with pytest.raises(ValueError, match="allow_missing_inputs"):
        PerturbationSpec(name="m", kind="set_missing", features=("x",), magnitude=0.1)
    with pytest.raises(ValueError, match="magnitude"):
        PerturbationSpec(name="m", kind="gaussian_noise", features=("x",))
    with pytest.raises(ValueError, match="together"):
        PerturbationSpec(name="m", kind="gaussian_noise", features=("x",), magnitude=0.1, max_metric_drop=0.1)
    spec = PerturbationSpec(name="m", kind="gaussian_noise", features=("ghost",), magnitude=0.1)
    with pytest.raises(MLConfigurationError, match="ghost"):
        Evaluator(EvaluationConfig(task="regression")).evaluate(
            regression_data(), EstimatorAdapter(Doubler(), model_id="d"), robustness=[spec])


def test_precomputed_predictions_cannot_be_perturbed() -> None:
    data = regression_data()
    spec = PerturbationSpec(name="n", kind="gaussian_noise", features=("x",), magnitude=0.1)
    with pytest.raises(MLConfigurationError, match="precomputed"):
        Evaluator(EvaluationConfig(task="regression")).evaluate(
            data, PrecomputedAdapter(list(data.frame["y"])), robustness=[spec])


# --- repeatability ----------------------------------------------------------------------------


def test_a_deterministic_model_is_repeatable() -> None:
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        classification_data(), EstimatorAdapter(Threshold(), model_id="t"), repeatability=RepeatabilityConfig(runs=3))
    assert report.repeatability.passed and report.repeatability.max_prediction_difference == 0.0


def test_a_random_model_fails_exact_repeatability_but_passes_with_a_tolerance() -> None:
    rng = np.random.default_rng(0)

    def noisy(X):
        return 2.0 * X["x"].to_numpy() + rng.normal(0.0, 0.01, len(X))

    config = EvaluationConfig(task="regression", metrics=("mae",))
    data = regression_data()
    strict = Evaluator(config).evaluate(data, FunctionAdapter(noisy, model_id="n"), repeatability=RepeatabilityConfig())
    assert not strict.repeatability.passed and not strict.passed
    loose = Evaluator(config).evaluate(
        data, FunctionAdapter(noisy, model_id="n"), repeatability=RepeatabilityConfig(stochastic=True, atol=0.1))
    assert loose.repeatability.passed and loose.repeatability.stochastic
    with pytest.raises(ValueError, match="atol"):
        RepeatabilityConfig(stochastic=True)


# --- subgroups and custom checks --------------------------------------------------------------


def test_subgroup_metrics_report_counts_and_insufficient_groups() -> None:
    data = classification_data()
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        data, EstimatorAdapter(Threshold(cut=2.0), model_id="t"), subgroups=SubgroupConfig(column="group", min_group_size=10))
    big, small = report.subgroups
    assert (big.group, big.n_samples, big.insufficient) == ("a", 36, False) and "accuracy" in big.metrics
    assert (small.group, small.n_samples, small.insufficient, small.metrics) == ("b", 4, True, {})
    assert any(f.code == "insufficient_subgroup" for f in report.warnings)


def test_a_continuous_grouping_column_is_refused() -> None:
    data = regression_data()
    report = Evaluator(EvaluationConfig(task="regression", max_groups=5)).evaluate(
        data, EstimatorAdapter(Doubler(), model_id="d"), subgroups=SubgroupConfig(column="x"))
    assert [f.code for f in report.errors] == ["too_many_groups"]


def test_custom_checks_plug_in_and_can_fail_the_report() -> None:
    class AtLeastTwoGroups:
        name = "group-coverage"

        def run(self, context: CheckContext) -> CheckResult:
            groups = context.dataset.frame["group"].nunique()
            return CheckResult(name=self.name, passed=groups >= 3, summary=f"{groups} group(s)")

    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        classification_data(), EstimatorAdapter(Threshold(), model_id="t"), checks=[AtLeastTwoGroups()])
    assert report.checks[0].passed is False and not report.passed
    assert "group-coverage" in report.failure_reasons[0]


def test_reports_are_reproducible() -> None:
    config = EvaluationConfig(task="classification", metrics=("accuracy", "log_loss"), positive_label=1)
    run = lambda: Evaluator(config).evaluate(classification_data(), EstimatorAdapter(Threshold(), model_id="t"))  # noqa: E731
    a, b = run(), run()
    assert a.evaluation_id == b.evaluation_id and a.metrics == b.metrics
    assert a.reproducibility == b.reproducibility and a.reproducibility["config_sha256"]
