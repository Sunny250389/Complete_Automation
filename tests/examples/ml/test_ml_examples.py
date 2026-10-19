"""Runnable examples of the ML module, written the way a team would write model quality tests.

These tests exercise *example* models on small synthetic or committed data; they demonstrate usage and
are not tests of the evaluation framework itself (see tests/unit/ml and tests/integration/ml).
Everything is deterministic and offline.
"""

import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression

from complete_automation.ml import (
    BaselineRule,
    ColumnSpec,
    DatasetSchema,
    EstimatorAdapter,
    EvaluationConfig,
    PerturbationSpec,
    PrecomputedAdapter,
    RepeatabilityConfig,
    ThresholdRule,
)
from complete_automation.ml import fixtures

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]


# --- precomputed predictions from a file -----------------------------------------------------


@pytest.fixture
def binary_file(ml_dataset_loader):
    schema = DatasetSchema(columns={
        "x1": ColumnSpec(dtype="numeric"), "x2": ColumnSpec(dtype="numeric"),
        "label": ColumnSpec(dtype="integer", allowed_values=(0, 1)),
        "p_pos": ColumnSpec(dtype="numeric", min=0.0, max=1.0),
    })
    return ml_dataset_loader.load(
        "examples/binary_predictions.csv", features=["x1", "x2"], target="label",
        dataset_id="binary-predictions", version="1", schema=schema)


@pytest.fixture
def precomputed(binary_file):
    return PrecomputedAdapter.from_frame(
        binary_file.frame, "prediction", probability_columns={0: "p_neg", 1: "p_pos"},
        model_id="example-model", version="1")


def test_precomputed_predictions_meet_the_quality_gate(binary_file, precomputed, ml_evaluate) -> None:
    config = EvaluationConfig(
        task="classification", positive_label=1,
        thresholds=(ThresholdRule.at_least("accuracy", 0.80), ThresholdRule.at_least("recall[1]", 0.80),
                    ThresholdRule.at_least("roc_auc", 0.90)))
    report = ml_evaluate(binary_file, precomputed, config, assert_passed=True)
    assert report.n_samples == 12 and report.value("accuracy") == pytest.approx(10 / 12)


def test_the_dataset_satisfies_its_schema_contract(binary_file, precomputed, ml_evaluate) -> None:
    report = ml_evaluate(binary_file, precomputed, EvaluationConfig(task="classification", metrics=("accuracy",)))
    assert report.errors == ()


def test_the_model_produces_valid_probability_distributions(binary_file, precomputed, ml_evaluate) -> None:
    report = ml_evaluate(binary_file, precomputed, EvaluationConfig(task="classification", metrics=("log_loss",)))
    assert not [f for f in report.errors if f.code.startswith(("invalid_probability", "probabilit"))]
    assert report.value("log_loss") < 0.7  # better than guessing 50/50 (ln 2 = 0.693)


def test_metrics_stay_within_limits_of_the_stored_baseline(binary_file, precomputed, ml_evaluate, ml_load_baseline) -> None:
    config = EvaluationConfig(task="classification", positive_label=1, metrics=("accuracy", "f1_binary", "recall_binary"))
    report = ml_evaluate(
        binary_file, precomputed, config, baseline=ml_load_baseline("example_binary.json"),
        baseline_rules=[BaselineRule(metric="accuracy", max_drop=0.02),
                        BaselineRule(metric="recall_binary", max_relative_drop=0.05)],
        assert_passed=True)
    assert report.baseline.passed


# --- an in-memory scikit-learn model ----------------------------------------------------------


def test_a_logistic_regression_classifier(ml_evaluate) -> None:
    data = fixtures.binary_classification()
    model = EstimatorAdapter(LogisticRegression().fit(data.X, data.y), model_id="logreg", version="demo")
    config = EvaluationConfig(
        task="classification", positive_label=1,
        thresholds=(ThresholdRule.at_least("f1_binary", 0.85), ThresholdRule.at_least("recall[1]", 0.85)),
        metrics=("accuracy", "f1_binary", "per_class", "roc_auc"))
    report = ml_evaluate(
        data, model, config,
        robustness=[PerturbationSpec(name="noise-5pct", kind="gaussian_noise", features=("x1", "x2"), magnitude=0.05,
                                     seed=7, gate_metric="accuracy", max_metric_drop=0.10)],
        repeatability=RepeatabilityConfig(runs=3), assert_passed=True)
    assert report.robustness[0].passed and report.repeatability.passed


def test_a_multiclass_classifier(ml_evaluate) -> None:
    data = fixtures.multiclass_classification()
    model = EstimatorAdapter(LogisticRegression().fit(data.X, data.y), model_id="multi")
    config = EvaluationConfig(task="classification", labels=("a", "b", "c"),
                              thresholds=(ThresholdRule.at_least("f1", 0.95),))
    ml_evaluate(data, model, config, assert_passed=True)


def test_a_regression_model_stays_within_its_error_limit(ml_evaluate) -> None:
    data = fixtures.regression()
    model = EstimatorAdapter(LinearRegression().fit(data.X, data.y), model_id="linear")
    config = EvaluationConfig(task="regression", thresholds=(ThresholdRule.at_most("rmse", 1.0),
                                                             ThresholdRule.at_least("r2", 0.95)))
    ml_evaluate(data, model, config, assert_passed=True)


def test_a_clustering_model_separates_the_groups(ml_evaluate) -> None:
    data = fixtures.clustering()
    model = EstimatorAdapter(KMeans(n_clusters=3, n_init=10, random_state=0).fit(data.X), model_id="kmeans")
    config = EvaluationConfig(task="clustering", thresholds=(ThresholdRule.at_least("silhouette", 0.5),
                                                             ThresholdRule.at_most("davies_bouldin", 0.5),
                                                             ThresholdRule.at_least("adjusted_rand_index", 0.9)))
    ml_evaluate(data, model, config, assert_passed=True)
