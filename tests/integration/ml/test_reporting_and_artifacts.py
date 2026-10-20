"""Reports through the shared-core artifact store and result recorder, with real scikit-learn models."""

import csv
import hashlib
import json

import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.redaction import Redactor
from complete_automation.ml import (
    Baseline,
    BaselineRule,
    EstimatorAdapter,
    EvaluationConfig,
    Evaluator,
    TaskType,
    ThresholdRule,
    compare_to_baseline,
)
from complete_automation.ml import fixtures
from complete_automation.ml.reporting import publish_report, record_result

pytestmark = [pytest.mark.ml, pytest.mark.integration, pytest.mark.cauto_module("ml")]


def classification_report(**config):
    data = fixtures.binary_classification()
    model = LogisticRegression().fit(data.X, data.y)
    cfg = EvaluationConfig(task="classification", positive_label=1, **config)
    return data, Evaluator(cfg).evaluate(data, EstimatorAdapter(model, model_id="logreg", version="1"))


def test_a_report_is_stored_as_json_and_csv_artifacts(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    _, report = classification_report(thresholds=(ThresholdRule.at_least("accuracy", 0.5),))
    recorder = ResultRecorder()

    published, refs = publish_report(report, store, recorder=recorder)

    names = sorted(ref.name for ref in refs)
    assert names == sorted([f"evaluation-{report.evaluation_id}.json", f"metrics-{report.evaluation_id}.csv",
                            f"confusion-matrix-{report.evaluation_id}.csv"])
    assert recorder.artifacts == list(refs)
    for ref in refs:
        path = store.resolve(ref)
        assert path.parent.name == "ml" and hashlib.sha256(path.read_bytes()).hexdigest() == ref.sha256
    stored = json.loads(store.resolve(next(r for r in refs if r.name.endswith(".json"))).read_text())
    assert stored["model"] == {"model_id": "logreg", "version": "1", "kind": "estimator", "class_name": "LogisticRegression"}
    assert stored["dataset"]["dataset_id"] == "synthetic-binary" and stored["task"] == "classification"
    assert stored["n_samples"] == 200 and stored["passed"] is True
    assert stored["gates"]["outcomes"][0]["metric"] == "accuracy"
    assert {a["name"] for a in stored["artifacts"]} == {n for n in names if n.endswith(".csv")}
    assert published.artifacts and report.artifacts == ()


def test_csv_summaries_are_tabular_and_complete(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    _, report = classification_report()
    _, refs = publish_report(report, store)
    metrics_text = store.resolve(next(r for r in refs if r.name.startswith("metrics-"))).read_text(encoding="utf-8")
    by_name = {row["metric"]: row for row in csv.DictReader(metrics_text.splitlines())}
    assert float(by_name["accuracy"]["value"]) == pytest.approx(report.value("accuracy"))
    assert by_name["confusion_matrix"]["value"] == ""
    matrix_text = store.resolve(next(r for r in refs if r.name.startswith("confusion"))).read_text(encoding="utf-8")
    matrix = list(csv.reader(matrix_text.splitlines()))
    assert matrix[0] == ["true\\predicted", "0", "1"] and sum(int(v) for row in matrix[1:] for v in row[1:]) == 200


def test_reports_contain_no_raw_rows_and_pass_through_the_redactor(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    data, report = classification_report()
    _, refs = publish_report(report, store, redactor=Redactor())
    text = store.resolve(next(r for r in refs if r.name.endswith(".json"))).read_text()
    assert str(round(float(data.frame["x1"].iloc[0]), 4)) not in text
    assert "token" not in text.lower() or "**" in text


def test_results_carry_gate_verdicts_as_metrics_and_a_summary() -> None:
    _, report = classification_report(thresholds=(ThresholdRule.at_least("accuracy", 0.99999),
                                                  ThresholdRule.at_least("recall[1]", 0.1)))
    recorder = ResultRecorder()
    record_result(recorder, report)
    metrics = {m.name: m for m in recorder.metrics}
    assert metrics["ml.accuracy"].threshold == 0.99999 and metrics["ml.accuracy"].passed is report.gates.outcomes[0].passed
    assert metrics["ml.recall[1]"].passed is True
    assert metrics["ml.f1"].passed is None  # no rule, no verdict
    summary = recorder.metadata["ml_evaluation"]
    assert summary["passed"] is report.passed and summary["n_samples"] == 200 and summary["model"]["model_id"] == "logreg"


def test_unavailable_gated_metrics_are_recorded_as_such_never_as_numbers() -> None:
    data = fixtures.regression()
    model = LinearRegression().fit(data.X, data.y)
    data = data.__class__.from_frame(data.frame.assign(y=0.0), "zeros", features=["x1", "x2"], target="y")
    report = Evaluator(EvaluationConfig(task="regression", metrics=("mape", "rmse"),
                                        thresholds=(ThresholdRule.at_most("mape", 0.5),))).evaluate(
        data, EstimatorAdapter(model, model_id="lin"))
    recorder = ResultRecorder()
    record_result(recorder, report)
    mape = next(m for m in recorder.metrics if m.name == "ml.mape")
    assert mape.value == "undefined" and mape.passed is False and "zero" in mape.metadata["reason"]


def test_baseline_round_trip_through_artifacts_and_comparison(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    _, report = classification_report()
    ref = Baseline.from_report(report).save(store, "baseline.json")

    adopted = tmp_path / "baselines"
    adopted.mkdir()
    (adopted / "m.json").write_text(store.resolve(ref).read_text(), encoding="utf-8")
    baseline = Baseline.load(adopted, "m.json")

    assert baseline.metrics["accuracy"] == report.value("accuracy") and "confusion_matrix" not in baseline.metrics
    unchanged = compare_to_baseline(baseline, report.metrics, [BaselineRule(metric="accuracy", max_drop=0.0)],
                                    task=TaskType.CLASSIFICATION)
    assert unchanged.passed and all(c.absolute_difference in (0.0, None) for c in unchanged.comparisons)

    worse = Baseline(**{**baseline.model_dump(), "metrics": {**baseline.metrics, "accuracy": 1.0}})
    gate = compare_to_baseline(worse, report.metrics, [BaselineRule(metric="accuracy", max_drop=0.0)],
                               task=TaskType.CLASSIFICATION)
    assert report.value("accuracy") < 1.0 and not gate.passed


def test_the_evaluator_accepts_a_baseline_and_flags_different_data() -> None:
    data, first = classification_report()
    baseline = Baseline.from_report(first)
    other = fixtures.binary_classification(seed=5)
    model = LogisticRegression().fit(other.X, other.y)
    report = Evaluator(EvaluationConfig(task="classification", positive_label=1, metrics=("accuracy",))).evaluate(
        other, EstimatorAdapter(model, model_id="logreg"), baseline=baseline,
        baseline_rules=[BaselineRule(metric="accuracy", max_drop=0.5)])
    assert report.baseline.passed and any(f.code == "baseline_dataset_differs" for f in report.warnings)
    assert data.metadata.fingerprint != other.metadata.fingerprint


def test_estimators_for_every_task_type_evaluate_end_to_end() -> None:
    multi = fixtures.multiclass_classification()
    model = LogisticRegression().fit(multi.X, multi.y)
    report = Evaluator(EvaluationConfig(task="classification")).evaluate(multi, EstimatorAdapter(model, model_id="m"))
    assert report.value("accuracy") > 0.9 and report.metric("roc_auc").status.value == "computed"
    assert report.metric("confusion_matrix").detail["labels"] == ["a", "b", "c"]

    reg = fixtures.regression()
    report = Evaluator(EvaluationConfig(task="regression")).evaluate(
        reg, EstimatorAdapter(LinearRegression().fit(reg.X, reg.y), model_id="lin"))
    assert report.value("r2") > 0.99 and report.value("rmse") < 1.0

    clusters = fixtures.clustering()
    report = Evaluator(EvaluationConfig(task="clustering")).evaluate(
        clusters, EstimatorAdapter(KMeans(3, n_init=10, random_state=0).fit(clusters.X), model_id="km"))
    assert report.value("silhouette") > 0.7 and report.value("adjusted_rand_index") == pytest.approx(1.0)
