"""Every module in one pytest session, reported through the one shared-core report.

A generated project registers the core, API, database, ML and LLM evaluation plugins
together (as tests/conftest.py registers the core and UI plugins) and runs passing and
failing tests of each module. The single JSON report must keep the modules apart, keep
their own metrics, and tell an assertion from a quality gate, an unavailable service and
an opt-in skip. The models and the HTTP service are scripted transports: this proves the
integration, not any model's quality (tests/integration/llm_evaluation/test_ollama_live.py
calls a real model).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

for _extra in ("httpx", "jsonschema", "sqlalchemy", "numpy", "pandas", "sklearn"):
    pytest.importorskip(_extra)

pytestmark = [pytest.mark.integration, pytest.mark.cross_module, pytest.mark.cauto_module("cross_module")]

PLUGINS = [
    "complete_automation.core.pytest_plugin",
    "complete_automation.api.pytest_plugin",
    "complete_automation.database.pytest_plugin",
    "complete_automation.ml.pytest_plugin",
    "complete_automation.llm_evaluation.pytest_plugin",
]

API_TESTS = '''
import httpx, pytest
from complete_automation.api.assertions import assert_status

pytestmark = pytest.mark.cauto_module("api")


def client(api_client_factory):
    def handler(request):
        return httpx.Response(200 if request.url.path == "/ok" else 404, json={"path": request.url.path})
    return api_client_factory(base_url="http://service.test", transport=httpx.MockTransport(handler))


def test_api_passes(api_client_factory):
    assert_status(client(api_client_factory).get("/ok"), 200)


def test_api_assertion_fails(api_client_factory):
    assert_status(client(api_client_factory).get("/missing"), 200)
'''

DATABASE_TESTS = '''
import pytest
from complete_automation.database.validations import assert_row_count

pytestmark = pytest.mark.cauto_module("database")


def test_database_passes(example_session):
    assert example_session.scalar("SELECT count(*) FROM customers") > 0


def test_database_assertion_fails(example_session):
    assert_row_count(example_session, "customers", 999)
'''

ML_TESTS = '''
import pytest
from sklearn.linear_model import LogisticRegression
from complete_automation.ml import EstimatorAdapter, EvaluationConfig, ThresholdRule, fixtures

pytestmark = pytest.mark.cauto_module("ml")


@pytest.fixture
def model():
    data = fixtures.binary_classification()
    return data, EstimatorAdapter(LogisticRegression().fit(data.X, data.y), model_id="logreg")


def gate(minimum):
    return EvaluationConfig(task="classification", positive_label=1, metrics=("accuracy", "f1_binary"),
                            thresholds=(ThresholdRule.at_least("accuracy", minimum),))


def test_ml_gate_passes(model, ml_evaluate):
    ml_evaluate(*model, gate(0.5), assert_passed=True)


def test_ml_gate_fails(model, ml_evaluate):
    ml_evaluate(*model, gate(0.9999999), assert_passed=True)
'''

LLM_TESTS = '''
import httpx, pytest
from complete_automation.llm_evaluation import (
    CheckSpec, CriterionSpec, EvaluationConfig, JudgeConfig, ModelConfig, OllamaClient, TestCase, ThresholdRule)

pytestmark = pytest.mark.cauto_module("llm_evaluation")


def ollama(*texts, error=None):
    replies = list(texts)

    def handler(request):
        if error is not None:
            raise error
        if request.url.path == "/api/tags":  # the client looks up the model digest; none is listed here
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": replies.pop(0)},
            "done": True, "done_reason": "stop", "eval_count": 4, "prompt_eval_count": 3})
    return OllamaClient(transport=httpx.MockTransport(handler))


CASE = TestCase(case_id="capital", prompt="Capital of France?", checks=(CheckSpec(type="contains", value="Paris"),))
CONFIG = EvaluationConfig(model=ModelConfig(model="target-model"),
                          suite_thresholds=(ThresholdRule(metric="case_pass_rate", operator="ge", value=1.0),))


def test_llm_passes(llm_evaluate):
    llm_evaluate([CASE], CONFIG, client=ollama("Paris"), assert_passed=True)


def test_llm_model_unreachable(llm_evaluate):
    llm_evaluate([CASE], CONFIG, client=ollama(error=httpx.ConnectError("refused")), assert_passed=True)


def test_llm_judge_failure_is_not_a_pass(llm_evaluate):
    judged = TestCase(case_id="judged", prompt="Capital of France?", reference="Paris",
                      criteria=(CriterionSpec(name="correctness", min_score=4),))
    config = EvaluationConfig(model=ModelConfig(model="target-model"),
                              judge=JudgeConfig(model=ModelConfig(model="judge-model")))
    llm_evaluate([judged], config, client=ollama("Paris"), judge_client=ollama("not a verdict"), assert_passed=True)


def test_llm_real_model_is_opt_in(llm_live_model):
    raise AssertionError("must not run without opt-in")
'''


@dataclass(frozen=True)
class Run:
    root: Path
    exit_code: int
    output: str
    report: dict[str, Any]

    def results(self) -> dict[str, dict[str, Any]]:
        return {result["name"]: result for result in self.report["results"]}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The generated project, run once in a separate pytest process for all the checks below."""
    root = tmp_path_factory.mktemp("unified_run")
    environments = root / "config" / "environments"
    environments.mkdir(parents=True)
    (environments / "dev.yaml").write_text(
        "modules:\n  ml:\n    data_root: data\n  llm_evaluation:\n    data_root: data\n    timeout_seconds: 2\n",
        encoding="utf-8",
    )
    (root / "data").mkdir()
    (root / "conftest.py").write_text(f"pytest_plugins = {PLUGINS!r}\n", encoding="utf-8")
    for name, source in (("api", API_TESTS), ("database", DATABASE_TESTS), ("ml", ML_TESTS), ("llm", LLM_TESTS)):
        (root / f"test_{name}_part.py").write_text(source, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("CAUTO_")}  # no inherited settings
    env["CAUTO_CONFIG_DIR"] = str(root / "config")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--cauto-report", "-rs", "-q"],
        cwd=root, env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    reports = list((root / "reports").glob("*/results.json"))
    assert len(reports) == 1, f"expected one report for the run, found {reports}\n{completed.stdout[-3000:]}"
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    return Run(root, completed.returncode, completed.stdout + completed.stderr, report)


def test_failures_in_any_module_fail_the_session(run: Run) -> None:
    summary = run.report["summary"]
    assert (summary["total"], summary["passed"], summary["skipped"]) == (10, 4, 1)
    assert summary["failed"] + summary["error"] == 5
    assert run.exit_code == pytest.ExitCode.TESTS_FAILED, run.output[-3000:]


def test_one_run_id_correlates_every_result_and_artifact(run: Run) -> None:
    report = run.report
    run_id = report["run"]["run_id"]
    assert {result["run_id"] for result in report["results"]} == {run_id}
    assert len({result["correlation_id"] for result in report["results"]}) == len(report["results"])
    artifact_root = run.root / "artifacts" / run_id
    paths = [artifact["path"] for result in report["results"] for artifact in result["artifacts"]]
    assert paths, "the evaluations and failures must have stored artifacts"
    assert all((artifact_root / path).is_file() for path in paths)
    assert {Path(path).parts[0] for path in paths} >= {"api", "database", "ml", "llm_evaluation"}


def test_each_module_is_counted_separately(run: Run) -> None:
    modules = run.report["summary"]["by_module"]
    assert set(modules) == {"api", "database", "ml", "llm_evaluation"}
    assert (modules["api"]["passed"], modules["api"]["failed"]) == (1, 1)
    assert (modules["database"]["passed"], modules["database"]["failed"]) == (1, 1)
    assert (modules["ml"]["passed"], modules["ml"]["failed"]) == (1, 1)
    llm = modules["llm_evaluation"]
    assert (llm["total"], llm["passed"], llm["skipped"]) == (4, 1, 1)
    assert llm["failed"] + llm["error"] == 2


def test_failure_categories_distinguish_assertions_gates_and_services(run: Run) -> None:
    results = run.results()
    assert results["test_api_assertion_fails"]["status"] == "failed"
    assert results["test_api_assertion_fails"]["failure"]["category"] == "assertion"
    assert results["test_database_assertion_fails"]["status"] == "failed"
    assert results["test_database_assertion_fails"]["failure"]["category"] == "assertion"
    assert results["test_ml_gate_fails"]["status"] == "failed"
    assert results["test_ml_gate_fails"]["failure"]["category"] == "threshold"
    assert results["test_llm_model_unreachable"]["status"] == "error"
    assert results["test_llm_model_unreachable"]["failure"]["category"] == "infrastructure"
    skipped = results["test_llm_real_model_is_opt_in"]
    assert skipped["status"] == "skipped" and "opt-in" in skipped["metadata"]["skip_reason"]


def test_a_judge_failure_never_becomes_a_pass(run: Run) -> None:
    judged = run.results()["test_llm_judge_failure_is_not_a_pass"]
    assert judged["status"] == "error"
    assert judged["failure"]["exception_type"].endswith("EvaluatorError")
    evaluation = judged["metadata"]["llm_evaluation"]
    assert (evaluation["counts"]["passed"], evaluation["counts"]["evaluator_error"]) == (0, 1)
    assert "judge output is not valid JSON" in evaluation["failure_reasons"][0]
    # the model under test and the judge stay distinguishable in the report
    assert evaluation["model"]["name"] == "target-model" and evaluation["judge"]["name"] == "judge-model"
    assert any(artifact["path"].startswith("llm_evaluation/responses") for artifact in judged["artifacts"])


def test_module_specific_metrics_are_kept_not_flattened(run: Run) -> None:
    results = run.results()
    ml = {metric["name"]: metric for metric in results["test_ml_gate_fails"]["metrics"]}
    assert ml["ml.accuracy"]["passed"] is False and ml["ml.accuracy"]["threshold"] == 0.9999999
    assert "ml.f1_binary" in ml
    llm = {metric["name"]: metric for metric in results["test_llm_passes"]["metrics"]}
    assert llm["llm.case_pass_rate"]["value"] == 1.0 and llm["llm.case_pass_rate"]["passed"] is True
    assert results["test_llm_passes"]["metadata"]["llm_evaluation"]["model"]["name"] == "target-model"
    api = {metric["name"] for metric in results["test_api_passes"]["metrics"]}
    database = {metric["name"] for metric in results["test_database_passes"]["metrics"]}
    assert "api.latency_ms" in api and "db.query_ms" in database
