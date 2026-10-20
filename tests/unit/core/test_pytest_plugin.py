"""Runs small pytest sessions in a subprocess to check fixtures, cleanup and reporting."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cauto_module("core")

PLUGIN_CONFTEST = 'pytest_plugins = ["complete_automation.core.pytest_plugin"]\n'


@pytest.fixture
def project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """An isolated project: its own config dir, no inherited CAUTO_* settings."""
    for name in ("CAUTO_ENVIRONMENT", "CAUTO_CONFIG_DIR", "CAUTO_EXECUTION__RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    config_dir = pytester.path / "config"
    (config_dir / "environments").mkdir(parents=True)
    (config_dir / "environments" / "dev.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("CAUTO_CONFIG_DIR", str(config_dir))
    pytester.makeconftest(PLUGIN_CONFTEST)
    return pytester


def report_of(project: pytest.Pytester) -> dict:
    reports = list((project.path / "reports").glob("*/results.json"))
    assert len(reports) == 1, reports
    return json.loads(reports[0].read_text(encoding="utf-8"))


def test_fixtures_scopes_and_isolation(project: pytest.Pytester) -> None:
    project.makepyfile(
        """
        from complete_automation.core.context import current_test_context

        seen = {}

        def test_first(framework_settings, run_context, test_context, artifact_store, result_recorder):
            assert framework_settings.environment == "dev"
            assert current_test_context() is test_context
            assert test_context.run_id == run_context.run_id
            ref = artifact_store.write_text("note.txt", "x")
            assert artifact_store.resolve(ref).is_file()
            result_recorder.add_metadata(step="first")
            seen["first"] = (run_context.run_id, test_context.correlation_id, id(framework_settings))

        def test_second(framework_settings, run_context, test_context, result_recorder):
            run_id, correlation_id, settings_id = seen["first"]
            assert run_context.run_id == run_id                   # session scope: shared
            assert id(framework_settings) == settings_id
            assert test_context.correlation_id != correlation_id  # function scope: fresh
            assert result_recorder.metadata == {}                 # no leakage between tests

        def test_no_context_without_the_fixture():
            assert current_test_context() is None
        """
    )

    result = project.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=3)
    assert not (project.path / "reports").exists()  # reporting is opt-in


def test_context_is_cleaned_up_after_a_failing_test(project: pytest.Pytester) -> None:
    project.makepyfile(
        """
        from complete_automation.core.context import current_test_context

        def test_fails(test_context):
            assert current_test_context() is test_context
            assert False, "deliberate"

        def test_after(test_context):
            assert current_test_context() is test_context

        def test_after_without_fixture():
            assert current_test_context() is None
        """
    )

    result = project.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=2, failed=1)


def test_report_classifies_every_outcome(project: pytest.Pytester) -> None:
    project.makepyfile(
        """
        import pytest

        @pytest.fixture
        def broken_setup():
            raise ConnectionError("db unreachable")

        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("cleanup failed")

        @pytest.mark.cauto_module("api")
        @pytest.mark.smoke
        def test_pass(result_recorder):
            result_recorder.add_metric("latency", 12, unit="ms")

        def test_assert_fail():
            assert 1 == 2

        def test_infra_error():
            raise ConnectionError("refused, password=hunter2")

        def test_setup_error(broken_setup):
            pass

        def test_teardown_error(broken_teardown):
            pass

        def test_skip():
            pytest.skip("not today")

        @pytest.mark.xfail(reason="known bug")
        def test_xfail():
            assert False

        def test_threshold(result_recorder):
            result_recorder.add_metric("faithfulness", 0.41, threshold=0.7, passed=False)
            assert False, "faithfulness below threshold"

        def test_incomplete(result_recorder):
            result_recorder.mark_incomplete("judge returned no score")
        """
    )

    result = project.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report")
    result.stdout.fnmatch_lines(["*complete_automation JSON report:*results.json*"])

    report = report_of(project)
    by_name = {r["name"]: r for r in report["results"]}

    assert len(report["results"]) == len(by_name) == 9  # exactly one result per test
    assert by_name["test_pass"]["status"] == "passed"
    assert by_name["test_pass"]["module"] == "api"
    assert by_name["test_pass"]["tags"] == ["smoke"]
    assert by_name["test_pass"]["metrics"][0] == {
        "name": "latency", "value": 12, "unit": "ms", "threshold": None, "passed": None, "metadata": {}
    }
    assert by_name["test_assert_fail"]["status"] == "failed"
    assert by_name["test_assert_fail"]["failure"]["category"] == "assertion"
    assert by_name["test_infra_error"]["status"] == "error"
    assert by_name["test_infra_error"]["failure"]["category"] == "infrastructure"
    assert "hunter2" not in json.dumps(by_name["test_infra_error"])
    assert by_name["test_setup_error"]["status"] == "error"
    assert by_name["test_teardown_error"]["status"] == "error"
    assert "cleanup failed" in by_name["test_teardown_error"]["failure"]["message"]
    assert by_name["test_skip"]["status"] == "skipped"
    assert by_name["test_skip"]["metadata"]["skip_reason"] == "not today"
    assert by_name["test_xfail"]["status"] == "skipped"
    assert by_name["test_xfail"]["metadata"]["skip_reason"] == "xfail: known bug"
    assert by_name["test_threshold"]["status"] == "failed"
    assert by_name["test_threshold"]["failure"]["category"] == "threshold"
    assert by_name["test_incomplete"]["status"] == "incomplete"
    assert report["summary"]["total"] == 9
    assert report["summary"]["error"] == 3
    assert report["run"]["finished_at"] is not None


def test_report_write_failure_is_not_swallowed(project: pytest.Pytester) -> None:
    (project.path / "blocker").write_text("a file, not a directory", encoding="utf-8")
    (project.path / "config" / "environments" / "dev.yaml").write_text(
        "reporting:\n  output_dir: blocker\n  json_enabled: true\n", encoding="utf-8"
    )
    project.makepyfile("def test_ok():\n    pass\n")

    result = project.runpytest_subprocess("-p", "no:cacheprovider")

    assert result.ret != 0
    assert "cannot write JSON report" in result.stdout.str() + result.stderr.str()


def test_invalid_configuration_is_a_usage_error(
    project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAUTO_RETRY__MAX_ATTEMPTS", "not-a-number-s3cr3t")
    project.makepyfile("def test_ok():\n    pass\n")

    result = project.runpytest_subprocess("-p", "no:cacheprovider")

    output = result.stdout.str() + result.stderr.str()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "retry.max_attempts" in output
    assert "not-a-number-s3cr3t" not in output


def test_environment_option_selects_the_file(project: pytest.Pytester) -> None:
    (project.path / "config" / "environments" / "test.yaml").write_text(
        "logging:\n  level: DEBUG\n", encoding="utf-8"
    )
    project.makepyfile(
        """
        def test_env(framework_settings):
            assert framework_settings.environment == "test"
            assert framework_settings.logging.level == "DEBUG"
        """
    )

    project.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-env", "test").assert_outcomes(passed=1)


def test_shared_run_id_from_configuration(project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAUTO_EXECUTION__RUN_ID", "shared-run-42")
    project.makepyfile("def test_ok():\n    pass\n")

    project.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report").assert_outcomes(passed=1)

    assert (Path(project.path) / "reports" / "shared-run-42" / "results.json").is_file()


def test_each_xdist_worker_writes_its_own_report(project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("xdist")
    monkeypatch.setenv("CAUTO_EXECUTION__RUN_ID", "parallel-run")
    project.makepyfile("import pytest\n\n@pytest.mark.parametrize('n', range(6))\ndef test_ok(n):\n    pass\n")

    project.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report", "-n", "2").assert_outcomes(passed=6)

    files = sorted(p.name for p in (Path(project.path) / "reports" / "parallel-run").iterdir())
    assert files == ["results-gw0.json", "results-gw1.json"]


def test_a_session_started_from_a_worker_is_not_itself_a_worker(
    project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")  # inherited by any pytest a worker's test starts
    monkeypatch.setenv("CAUTO_EXECUTION__RUN_ID", "nested-run")
    project.makepyfile("def test_ok():\n    pass\n")

    project.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report").assert_outcomes(passed=1)

    assert [p.name for p in (Path(project.path) / "reports" / "nested-run").iterdir()] == ["results.json"]
