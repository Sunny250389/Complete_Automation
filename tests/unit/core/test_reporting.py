import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from complete_automation.core.context import RunContext
from complete_automation.core.exceptions import ReportingError
from complete_automation.core.reporting import JsonReporter, Reporter, summarize
from complete_automation.core.results import ErrorCategory, FailureInfo, MetricResult, Status, TestResult

START = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)


def result(name: str, status: Status, module: str = "api", seconds: float = 1.0, **extra) -> TestResult:
    if status in (Status.FAILED, Status.ERROR) and "failure" not in extra:
        extra["failure"] = FailureInfo(message="boom", category=ErrorCategory.ASSERTION)
    return TestResult(
        run_id="run-1",
        test_id=f"tests/{module}/test_x.py::{name}",
        name=name,
        module=module,
        status=status,
        started_at=START,
        finished_at=START + timedelta(seconds=seconds),
        **extra,
    )


def sample() -> list[TestResult]:
    return [
        result("a", Status.PASSED, metrics=(MetricResult(name="f1", value=0.875),)),
        result("b", Status.FAILED, seconds=2.0),
        result("c", Status.ERROR, module="db"),
        result("d", Status.SKIPPED, module="db", seconds=0.0),
        result("e", Status.INCOMPLETE, module="llm", seconds=0.5),
    ]


def test_summary_counts_by_status_and_module() -> None:
    summary = summarize(sample())

    assert (summary.total, summary.passed, summary.failed, summary.skipped, summary.error, summary.incomplete) == (
        5, 1, 1, 1, 1, 1,
    )
    assert summary.duration_seconds == 4.5
    assert summary.metric_count == 1
    assert sorted(summary.by_module) == ["api", "db", "llm"]
    assert summary.by_module["db"].error == 1 and summary.by_module["db"].skipped == 1


def test_empty_run_summary() -> None:
    assert summarize([]).total == 0


def test_json_report_document(tmp_path: Path) -> None:
    run = RunContext.start("test", run_id="run-1")
    reporter = JsonReporter(tmp_path / "out" / "results.json")
    reporter.start_run(run)
    for item in sample():
        reporter.publish(item)

    path = reporter.finalize(run.finish())
    document = json.loads(path.read_text(encoding="utf-8"))

    assert set(document) == {"schema_version", "run", "summary", "results"}
    assert document["run"]["run_id"] == "run-1" and document["run"]["finished_at"]
    assert document["summary"]["total"] == 5
    assert [r["name"] for r in document["results"]] == ["a", "b", "c", "d", "e"]
    assert document["results"][0]["metrics"][0]["value"] == 0.875  # passed through unchanged
    assert not list(path.parent.glob(".*.tmp"))


def test_report_output_is_deterministic(tmp_path: Path) -> None:
    run = RunContext.start("test", run_id="run-1")
    texts = []
    for name in ("one.json", "two.json"):
        reporter = JsonReporter(tmp_path / name)
        for item in sample():
            reporter.publish(item)
        texts.append(reporter.finalize(run).read_text(encoding="utf-8"))

    assert texts[0] == texts[1]


def test_lifecycle_misuse_raises(tmp_path: Path) -> None:
    reporter = JsonReporter(tmp_path / "r.json")
    reporter.finalize()

    with pytest.raises(ReportingError, match="already finalized"):
        reporter.publish(result("a", Status.PASSED))
    with pytest.raises(ReportingError, match="already finalized"):
        reporter.finalize()


def test_only_test_results_are_accepted(tmp_path: Path) -> None:
    with pytest.raises(ReportingError, match="expected TestResult"):
        JsonReporter(tmp_path / "r.json").publish({"status": "passed"})  # type: ignore[arg-type]


def test_write_failures_are_raised_not_swallowed(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed", encoding="utf-8")
    reporter = JsonReporter(blocker / "results.json")
    reporter.publish(result("a", Status.PASSED))

    with pytest.raises(ReportingError, match="cannot write JSON report") as raised:
        reporter.finalize()
    assert isinstance(raised.value.__cause__, OSError)


def test_failed_cleanup_does_not_mask_the_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux raises NotADirectoryError from unlink() here; Windows does not. Force it on any OS."""

    def failing_replace(*args: object) -> None:
        raise PermissionError("target is locked")

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        raise NotADirectoryError("parent is a file")

    reporter = JsonReporter(tmp_path / "results.json")
    reporter.publish(result("a", Status.PASSED))
    monkeypatch.setattr("complete_automation.core.reporting.os.replace", failing_replace)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(ReportingError, match="cannot write JSON report") as raised:
        reporter.finalize()
    assert isinstance(raised.value.__cause__, PermissionError)


def test_report_is_redacted(tmp_path: Path) -> None:
    reporter = JsonReporter(tmp_path / "r.json")
    reporter.publish(
        result(
            "a",
            Status.FAILED,
            failure=FailureInfo(message="login failed: password=hunter2", category=ErrorCategory.ASSERTION),
            metadata={"api_token": "tok-123", "prompt_tokens": 42},
        )
    )

    text = reporter.finalize().read_text(encoding="utf-8")

    assert "hunter2" not in text and "tok-123" not in text
    assert '"prompt_tokens": 42' in text


def test_json_reporter_satisfies_the_reporter_protocol(tmp_path: Path) -> None:
    assert isinstance(JsonReporter(tmp_path / "r.json"), Reporter)
