"""The ML fixtures inside real pytest sessions: genuine failures, shared-core reports, data roots."""

import json

import pytest

pytestmark = [pytest.mark.ml, pytest.mark.integration, pytest.mark.cauto_module("ml")]

CONFTEST = 'pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.ml.pytest_plugin"]\n'


@pytest.fixture
def project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    for name in ("CAUTO_ENVIRONMENT", "CAUTO_CONFIG_DIR", "CAUTO_EXECUTION__RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    config_dir = pytester.path / "config"
    (config_dir / "environments").mkdir(parents=True)
    (config_dir / "environments" / "dev.yaml").write_text("modules:\n  ml:\n    data_root: data\n", encoding="utf-8")
    (pytester.path / "data").mkdir()
    monkeypatch.setenv("CAUTO_CONFIG_DIR", str(config_dir))
    pytester.makeconftest(CONFTEST)
    return pytester


def results(project):
    (path,) = list((project.path / "reports").glob("*/results.json"))
    return {r["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))["results"]}


TEST_FILE = '''
import pytest
from sklearn.linear_model import LogisticRegression
from complete_automation.ml import EstimatorAdapter, EvaluationConfig, ThresholdRule, fixtures


@pytest.fixture
def setup():
    data = fixtures.binary_classification()
    return data, EstimatorAdapter(LogisticRegression().fit(data.X, data.y), model_id="logreg")


def config(minimum):
    return EvaluationConfig(task="classification", positive_label=1, metrics=("accuracy", "f1_binary"),
                            thresholds=(ThresholdRule.at_least("accuracy", minimum),))


def test_meets_its_threshold(setup, ml_evaluate):
    report = ml_evaluate(*setup, config(0.5), assert_passed=True)
    assert report.passed


def test_misses_its_threshold(setup, ml_evaluate):
    ml_evaluate(*setup, config(0.9999999), assert_passed=True)


def test_report_is_inspectable_without_failing(setup, ml_evaluate):
    report = ml_evaluate(*setup, config(0.9999999))
    assert not report.passed and report.gates.threshold_failures[0].metric == "accuracy"


def test_invalid_input_is_a_validation_failure(ml_evaluate, setup):
    import pandas as pd
    from complete_automation.ml import Dataset, PrecomputedAdapter
    data = Dataset.from_frame(pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]}), "tiny", features=["x"], target="y")
    ml_evaluate(data, PrecomputedAdapter([0, None]), config(0.5), assert_passed=True)
'''


def test_threshold_failures_are_genuine_pytest_failures_with_diagnostics(project: pytest.Pytester) -> None:
    project.makepyfile(TEST_FILE)

    result = project.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report")

    result.assert_outcomes(passed=2, failed=2)
    result.stdout.fnmatch_lines(["*QualityGateError: ML evaluation failed*",
                                 "*threshold failed: accuracy observed *, required >= 0.9999999*"])
    result.stdout.fnmatch_lines(["*ValidationFailedError: ML evaluation failed*", "*1 prediction(s) are missing*"])

    recorded = results(project)
    failed = recorded["test_misses_its_threshold"]
    assert failed["status"] == "failed" and failed["failure"]["category"] == "threshold"
    metric = next(m for m in failed["metrics"] if m["name"] == "ml.accuracy")
    assert metric["passed"] is False and metric["threshold"] == 0.9999999
    assert failed["metadata"]["ml_evaluation"]["passed"] is False
    assert {a["name"].split("-")[0] for a in failed["artifacts"]} == {"evaluation", "metrics"}
    assert recorded["test_meets_its_threshold"]["status"] == "passed"
    assert recorded["test_report_is_inspectable_without_failing"]["status"] == "passed"
    assert recorded["test_invalid_input_is_a_validation_failure"]["failure"]["category"] == "test_data"
    assert recorded["test_invalid_input_is_a_validation_failure"]["status"] == "error"


def test_artifacts_are_written_under_the_run_directory(project: pytest.Pytester) -> None:
    project.makepyfile(TEST_FILE)
    project.runpytest_subprocess("-p", "no:cacheprovider", "-k", "meets")
    written = sorted(p.name.split("-")[0] for p in (project.path / "artifacts").rglob("ml/*"))
    assert written == ["evaluation", "metrics"]


def test_datasets_load_from_the_configured_data_root_only(project: pytest.Pytester) -> None:
    (project.path / "data" / "d.csv").write_text("x,label\n1.0,0\n2.0,1\n3.0,0\n4.0,1\n", encoding="utf-8")
    project.makepyfile(
        """
        import pytest
        from complete_automation.core.exceptions import TestDataError

        def test_loads(ml_dataset_loader, ml_settings):
            data = ml_dataset_loader.load("d.csv", features=["x"], target="label")
            assert data.metadata.n_rows == 4 and ml_settings.data_root.name == "data"

        def test_cannot_escape_the_root(ml_dataset_loader):
            with pytest.raises(TestDataError):
                ml_dataset_loader.load("../conftest.py", features=["x"])
        """
    )
    project.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=2)


def test_artifact_publication_can_be_switched_off(project: pytest.Pytester) -> None:
    (project.path / "config" / "environments" / "dev.yaml").write_text(
        "modules:\n  ml:\n    data_root: data\n    publish_artifacts: false\n", encoding="utf-8")
    project.makepyfile(TEST_FILE)
    project.runpytest_subprocess("-p", "no:cacheprovider", "-k", "meets").assert_outcomes(passed=1)
    assert not list((project.path / "artifacts").rglob("ml/*"))


def test_the_baseline_fixture_loads_from_the_baseline_directory(project: pytest.Pytester) -> None:
    (project.path / "config" / "environments" / "dev.yaml").write_text(
        "modules:\n  ml:\n    data_root: data\n    baseline_dir: baselines\n", encoding="utf-8")
    (project.path / "baselines").mkdir()
    (project.path / "baselines" / "b.json").write_text(json.dumps({
        "baseline_id": "b", "task": "classification", "model_id": "m", "dataset_id": "d",
        "dataset_fingerprint": "f", "metrics": {"accuracy": 0.9}}), encoding="utf-8")
    project.makepyfile(
        """
        def test_loads(ml_load_baseline):
            assert ml_load_baseline("b.json").metrics == {"accuracy": 0.9}
        """
    )
    project.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)
