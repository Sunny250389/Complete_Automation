"""Markdown summaries of JSON reports: counts per module, failures grouped by cause, skips by reason."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from complete_automation.core.artifacts import ArtifactRef
from complete_automation.core.context import RunContext
from complete_automation.core.exceptions import ErrorCategory
from complete_automation.core.reporting import JsonReporter
from complete_automation.core.results import FailureInfo, MetricResult, Status, TestResult
from complete_automation.core.summary import ReportError, load_reports, main, render_markdown

pytestmark = pytest.mark.cauto_module("core")

START = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)


def result(name: str, module: str, status: Status, category: ErrorCategory | None = None, **extra) -> TestResult:
    failure = FailureInfo(message=f"{name} | went wrong", category=category) if category else None
    return TestResult(
        run_id="run-1", test_id=f"tests/{module}/test_x.py::{name}", name=name, module=module, status=status,
        started_at=START, finished_at=START + timedelta(seconds=1.5), failure=failure, **extra,
    )


ARTIFACT = ArtifactRef(name="metrics.csv", path="ml/metrics.csv", media_type="text/csv", size_bytes=3,
                       sha256="0" * 64, created_at=START)
RESULTS = [
    result("test_ok", "api", Status.PASSED),
    result("test_status", "api", Status.FAILED, ErrorCategory.ASSERTION),
    result("test_gate", "ml", Status.FAILED, ErrorCategory.THRESHOLD, artifacts=(ARTIFACT,), metrics=(
        MetricResult(name="ml.accuracy", value=0.8, threshold=0.9, passed=False),
        MetricResult(name="ml.f1_binary", value=0.7),
    )),
    result("test_model_down", "llm_evaluation", Status.ERROR, ErrorCategory.INFRASTRUCTURE),
    result("test_bad_config", "database", Status.ERROR, ErrorCategory.CONFIGURATION),
    result("test_no_verdict", "llm_evaluation", Status.INCOMPLETE, ErrorCategory.UNKNOWN),
    result("test_live_1", "llm_evaluation", Status.SKIPPED, metadata={"skip_reason": "real-model tests are opt-in"}),
    result("test_live_2", "llm_evaluation", Status.SKIPPED, metadata={"skip_reason": "real-model tests are opt-in"}),
    result("test_pg", "database", Status.SKIPPED, metadata={"skip_reason": "PostgreSQL is not configured"}),
]  # fmt: skip


def write_report(path: Path, results: list[TestResult], run_id: str = "run-1") -> Path:
    reporter = JsonReporter(path)
    run = RunContext.start("test", run_id=run_id)
    reporter.start_run(run)
    for item in results:
        reporter.publish(item.model_copy(update={"run_id": run_id}))
    return reporter.finalize(run.finish())


def section(text: str, heading: str) -> str:
    return text.split(f"### {heading}", 1)[1].split("\n### ", 1)[0]


def test_counts_overall_and_per_module() -> None:
    text = render_markdown([{"run_id": "run-1", "environment": "test"}], RESULTS)
    assert "Run `run-1`, environment `test`" in text
    assert "| 9 | 1 | 2 | 2 | 3 | 1 | 13.5 |" in text
    assert "| api | 2 | 1 | 1 | 0 | 0 | 0 |" in text
    assert "| llm_evaluation | 4 | 0 | 0 | 1 | 2 | 1 |" in text


def test_failures_are_grouped_by_cause() -> None:
    text = render_markdown([], RESULTS)
    assert "test_gate" in section(text, "Quality-gate failures (1)")
    assert "test_status" in section(text, "Assertion failures (1)")
    assert "test_model_down" in section(text, "Service or infrastructure errors (1)")
    assert "test_bad_config" in section(text, "Other errors (1)")
    assert "test_no_verdict" in section(text, "Incomplete (no verdict) (1)")


def test_failed_metrics_and_artifacts_are_listed_and_passing_metrics_are_not() -> None:
    gate = section(render_markdown([], RESULTS), "Quality-gate failures (1)")
    assert "ml.accuracy=0.8 (threshold 0.9)" in gate and "ml.f1_binary" not in gate
    assert "ml/metrics.csv" in gate


def test_skips_are_grouped_by_reason() -> None:
    skipped = section(render_markdown([], RESULTS), "Skipped (3)")
    assert "| 2 | llm_evaluation | real-model tests are opt-in |" in skipped
    assert "| 1 | database | PostgreSQL is not configured |" in skipped


def test_table_cells_stay_on_one_line_with_pipes_escaped() -> None:
    text = render_markdown([], [result("test_status", "api", Status.FAILED, ErrorCategory.ASSERTION)])
    assert "test_status \\| went wrong" in text


def test_a_passing_run_has_no_failure_sections() -> None:
    text = render_markdown([], [result("test_ok", "api", Status.PASSED)])
    assert "###" not in text


def test_several_reports_are_combined(tmp_path: Path) -> None:
    first = write_report(tmp_path / "a" / "results.json", RESULTS[:3], run_id="run-a")
    second = write_report(tmp_path / "b" / "results.json", RESULTS[3:], run_id="run-b")
    runs, results = load_reports([first, second])
    assert [run["run_id"] for run in runs] == ["run-a", "run-b"]
    assert len(results) == len(RESULTS)


def test_the_command_line_appends_to_a_file_and_expands_patterns(tmp_path: Path) -> None:
    write_report(tmp_path / "r1" / "results.json", RESULTS)
    output = tmp_path / "summary.md"
    output.write_text("existing\n", encoding="utf-8")
    assert main([str(tmp_path / "*" / "results.json"), "--output", str(output), "--title", "Unit tests"]) == 0
    text = output.read_text(encoding="utf-8")
    assert text.startswith("existing\n## Unit tests")


def test_missing_or_invalid_reports_exit_with_status_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "*.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"run": {}, "results": [{"status": "nonsense"}]}', encoding="utf-8")
    assert main([str(bad)]) == 2
    assert "not a readable shared-core report" in capsys.readouterr().err
    with pytest.raises(ReportError):
        load_reports([tmp_path / "absent.json"])
