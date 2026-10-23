"""The test suite itself: results must not depend on the developer's shell, and skipped suites leave nothing behind.

Each check runs real tests of the suite in a separate pytest process under the condition
that used to break them.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cross_module, pytest.mark.cauto_module("cross_module")]

ROOT = Path(__file__).resolve().parents[2]
ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")  # CI sets FORCE_COLOR, which colors the inner run's report

# What docs/integration_and_ci.md tells a developer to export for live tests, plus common overrides.
EXPORTED_FOR_LIVE_TESTS = {
    "CAUTO_ENVIRONMENT": "test",
    "CAUTO_LOGGING__LEVEL": "DEBUG",
    "CAUTO_MODULES__API__BASE_URL": "https://api.example.test",
    "CAUTO_MODULES__LLM_EVALUATION__MODEL": "some-installed-model",
    "CAUTO_MODULES__LLM_EVALUATION__JUDGE_MODEL": "some-judge-model",
    "CAUTO_MODULES__LLM_EVALUATION__REQUIRE_MODEL": "true",
}

# Plugin self-tests whose generated projects used to inherit the variables above and fail.
SELF_TESTS = [
    "tests/unit/api/test_settings_and_reporting.py::test_fixtures_publish_api_results_through_the_core_plugins",
    "tests/unit/core/test_pytest_plugin.py::test_fixtures_scopes_and_isolation",
    "tests/integration/llm_evaluation/test_pytest_integration.py::test_live_without_a_model_name_skips_and_names_the_setting",
]


def run_pytest(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False,
    )


def clean_env(tmp_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("CAUTO_")}
    env["CAUTO_REPORTING__OUTPUT_DIR"] = str(tmp_path / "reports")  # the outer run reports nowhere in the repository
    env["CAUTO_ARTIFACTS__ROOT"] = str(tmp_path / "artifacts")
    return env


def test_plugin_self_tests_ignore_cauto_settings_exported_in_the_shell(tmp_path: Path) -> None:
    pytest.importorskip("openapi_spec_validator")  # the "api" extra
    pytest.importorskip("jsonschema")
    env = {**clean_env(tmp_path), **EXPORTED_FOR_LIVE_TESTS}

    completed = run_pytest(SELF_TESTS, env)

    assert completed.returncode == 0, completed.stdout[-3000:]


def test_the_legacy_deepeval_suite_without_a_judge_skips_and_creates_no_run_directory(tmp_path: Path) -> None:
    pytest.importorskip("deepeval")
    env = clean_env(tmp_path)
    env.pop("OPENAI_API_KEY", None)
    dotenv = ROOT / ".env"
    if dotenv.is_file() and "OPENAI_API_KEY" in dotenv.read_text(encoding="utf-8", errors="replace"):
        pytest.skip("a .env file configures the DeepEval judge here")
    reports = ROOT / "reports"
    before = {path.name for path in reports.glob("run_*")} if reports.is_dir() else set()

    completed = run_pytest(["tests/llm", "-rs"], env)

    after = {path.name for path in reports.glob("run_*")} if reports.is_dir() else set()
    assert completed.returncode == 0, completed.stdout[-3000:]
    output = ANSI_COLOR.sub("", completed.stdout)
    skipped = [line for line in output.splitlines() if line.startswith("SKIPPED")]
    assert len(skipped) == 2 and all("DeepEval judge not configured" in line for line in skipped), output
    assert after == before
