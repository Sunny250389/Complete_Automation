"""The LLM fixtures inside real pytest sessions: failures, skips, shared-core reports and artifacts.

The model here is a scripted HTTP transport (mocked). These tests prove the pytest integration, not
any real model; the real-model tests are in test_ollama_live.py.
"""

import json

import pytest

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.integration, pytest.mark.cauto_module("llm_evaluation")]

CONFTEST = 'pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.llm_evaluation.pytest_plugin"]\n'


@pytest.fixture
def project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    for name in ("CAUTO_ENVIRONMENT", "CAUTO_CONFIG_DIR", "CAUTO_EXECUTION__RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    config_dir = pytester.path / "config"
    (config_dir / "environments").mkdir(parents=True)
    (config_dir / "environments" / "dev.yaml").write_text(
        "reporting:\n  json_enabled: true\nmodules:\n  llm_evaluation:\n    data_root: data\n    timeout_seconds: 2\n",
        encoding="utf-8",
    )
    (pytester.path / "data").mkdir()
    monkeypatch.setenv("CAUTO_CONFIG_DIR", str(config_dir))
    pytester.makeconftest(CONFTEST)
    return pytester


def results(project):
    (path,) = list((project.path / "reports").glob("*/results.json"))
    return {r["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))["results"]}


MOCK = '''
import httpx, pytest
from complete_automation.llm_evaluation import (
    CheckSpec, EvaluationConfig, ModelConfig, OllamaClient, TestCase, ThresholdRule)

def client(text=None, error=None):
    def handler(request):
        if error is not None:
            raise error
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}, "done": True,
                                          "done_reason": "stop", "eval_count": 4, "prompt_eval_count": 3})
    return OllamaClient(transport=httpx.MockTransport(handler))

CASE = TestCase(case_id="c1", prompt="capital?", checks=(CheckSpec(type="contains", value="Paris"),))
CONFIG = EvaluationConfig(model=ModelConfig(model="mock-model"),
                          suite_thresholds=(ThresholdRule(metric="case_pass_rate", operator="ge", value=1.0),))
'''


def test_passing_failing_and_unavailable_model_are_reported_with_the_right_status(project) -> None:
    project.makepyfile(MOCK + '''
def test_passes(llm_evaluate):
    report = llm_evaluate([CASE], CONFIG, client=client("Paris"), assert_passed=True)
    assert report.passed

def test_threshold_failure(llm_evaluate):
    llm_evaluate([CASE], CONFIG, client=client("Rome"), assert_passed=True)

def test_model_unreachable(llm_evaluate):
    llm_evaluate([CASE], CONFIG, client=client(error=httpx.ConnectError("refused")), assert_passed=True)

def test_inspect_without_failing(llm_evaluate):
    report = llm_evaluate([CASE], CONFIG, client=client("Rome"))
    assert not report.passed and report.counts["failed"] == 1
''')
    outcome = project.runpytest("-p", "no:cacheprovider")
    outcome.assert_outcomes(passed=2, failed=2)
    data = results(project)
    assert data["test_passes"]["status"] == "passed"
    assert data["test_threshold_failure"]["status"] == "failed"
    assert data["test_threshold_failure"]["failure"]["category"] == "threshold"
    assert data["test_model_unreachable"]["failure"]["category"] == "infrastructure"
    assert "ModelInvocationError" in data["test_model_unreachable"]["failure"]["message"]


def test_results_carry_llm_metrics_metadata_and_artifacts(project) -> None:
    project.makepyfile(MOCK + '''
def test_it(llm_evaluate):
    llm_evaluate([CASE], CONFIG, client=client("Paris"), assert_passed=True)
''')
    project.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)
    result = results(project)["test_it"]
    metrics = {m["name"]: m for m in result["metrics"]}
    assert metrics["llm.case_pass_rate"]["value"] == 1.0 and metrics["llm.case_pass_rate"]["passed"] is True
    meta = result["metadata"]["llm_evaluation"]
    assert meta["model"]["name"] == "mock-model" and meta["counts"]["passed"] == 1
    names = sorted(a["name"] for a in result["artifacts"])
    assert [n.split("-")[0] for n in names] == ["cases", "evaluation", "metrics", "responses"]
    run_dirs = list((project.path / "artifacts").glob("*/llm_evaluation"))
    assert len(run_dirs) == 1 and len(list(run_dirs[0].iterdir())) == 4


def test_real_model_fixture_skips_unless_opted_in_and_configured(project) -> None:
    project.makepyfile('''
def test_needs_model(llm_live_model):
    raise AssertionError("must not run without opt-in")
''')
    outcome = project.runpytest("-p", "no:cacheprovider", "-rs")
    outcome.assert_outcomes(skipped=1)
    outcome.stdout.fnmatch_lines(["*real-model tests are opt-in*"])
    assert results(project)["test_needs_model"]["status"] == "skipped"


def test_live_without_a_model_name_skips_and_names_the_setting(project, monkeypatch) -> None:
    monkeypatch.setenv("CAUTO_MODULES__LLM_EVALUATION__LIVE", "true")
    project.makepyfile("def test_needs_model(llm_live_model):\n    raise AssertionError\n")
    outcome = project.runpytest("-p", "no:cacheprovider", "-rs")
    outcome.assert_outcomes(skipped=1)
    outcome.stdout.fnmatch_lines(["*no model configured*"])


def test_unreachable_endpoint_skips_or_fails_according_to_policy(project, monkeypatch) -> None:
    monkeypatch.setenv("CAUTO_MODULES__LLM_EVALUATION__LIVE", "true")
    monkeypatch.setenv("CAUTO_MODULES__LLM_EVALUATION__MODEL", "anything")
    monkeypatch.setenv("CAUTO_MODULES__LLM_EVALUATION__BASE_URL", "http://127.0.0.1:9")
    project.makepyfile("def test_needs_model(llm_live_model):\n    raise AssertionError('no model must mean no evaluation')\n")
    skipped = project.runpytest("-p", "no:cacheprovider", "-rs")
    skipped.assert_outcomes(skipped=1)
    skipped.stdout.fnmatch_lines(["*not usable*cannot connect to Ollama*"])

    monkeypatch.setenv("CAUTO_MODULES__LLM_EVALUATION__REQUIRE_MODEL", "true")
    failed = project.runpytest("-p", "no:cacheprovider")
    failed.assert_outcomes(errors=1)


def test_dataset_and_baseline_fixtures_are_confined_to_their_roots(project) -> None:
    (project.path / "data" / "cases.jsonl").write_text(
        json.dumps({"case_id": "a", "prompt": "q", "checks": [{"type": "not_empty"}]}) + "\n", encoding="utf-8"
    )
    project.makepyfile('''
import pytest
from complete_automation.llm_evaluation import LLMDatasetError

def test_loads(llm_load_dataset):
    assert llm_load_dataset("cases.jsonl").cases[0].case_id == "a"

def test_refuses_escape(llm_load_dataset):
    with pytest.raises(LLMDatasetError):
        llm_load_dataset("../outside.jsonl")
''')
    project.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=2)
