import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from complete_automation.core.artifacts import ArtifactRef
from complete_automation.core.results import (
    AttemptInfo,
    ErrorCategory,
    FailureInfo,
    MetricResult,
    Status,
    TestResult,
)

pytestmark = pytest.mark.cauto_module("core")

START = datetime(2026, 10, 16, 9, 0, 0, tzinfo=UTC)
END = START + timedelta(seconds=1.5)


def make(**overrides) -> TestResult:
    values = dict(
        run_id="run-1",
        test_id="tests/llm/test_x.py::test_answer[case-1]",
        name="test_answer[case-1]",
        module="llm",
        status=Status.PASSED,
        started_at=START,
        finished_at=END,
    )
    values.update(overrides)
    return TestResult(**values)


def artifact() -> ArtifactRef:
    return ArtifactRef(
        name="screenshot.png",
        path="ui/screenshot.png",
        media_type="image/png",
        size_bytes=10,
        sha256="0" * 64,
        created_at=END,
    )


def test_status_vocabulary_is_small_and_explicit() -> None:
    assert [s.value for s in Status] == ["passed", "failed", "skipped", "error", "incomplete"]


def test_minimal_result_and_duration() -> None:
    result = make()

    assert result.duration_seconds == 1.5
    assert result.failure is None and result.metrics == () and result.metadata == {}


def test_llm_and_ui_results_share_the_contract() -> None:
    llm = make(
        status=Status.FAILED,
        failure=FailureInfo(message="faithfulness 0.41 < 0.7", category=ErrorCategory.THRESHOLD),
        metrics=(
            MetricResult(name="faithfulness", value=0.41, threshold=0.7, passed=False),
            MetricResult(name="correctness", value=0.9, threshold=0.7, passed=True),
        ),
        metadata={"model": "llama3.1:8b", "judge": {"provider": "ollama", "model": "qwen2.5"}},
    )
    ui = make(module="ui", name="test_login", test_id="tests/ui/test_login.py::test_login", artifacts=(artifact(),))

    assert llm.metrics[0].passed is False
    assert ui.artifacts[0].media_type == "image/png"


def test_serialization_is_json_and_keeps_every_field() -> None:
    result = make(
        status=Status.ERROR,
        failure=FailureInfo(
            message="ConnectionError: refused",
            category=ErrorCategory.INFRASTRUCTURE,
            exception_type="builtins.ConnectionError",
        ),
        metrics=(MetricResult(name="latency", value=120, unit="ms"),),
        artifacts=(artifact(),),
        metadata={"nested": {"list": [1, 2.5, "x", None, True]}},
        attempt=AttemptInfo(attempt=2, max_attempts=3),
        tags=("smoke",),
        correlation_id="abc",
    )

    data = result.to_dict()
    text = json.dumps(data)

    assert set(data) == set(TestResult.model_fields) | {"duration_seconds"}
    assert data["status"] == "error"
    assert data["failure"]["category"] == "infrastructure"
    assert data["started_at"] == "2026-10-16T09:00:00Z"
    assert TestResult.from_dict(json.loads(text)) == result


def test_failed_and_error_results_need_failure_details() -> None:
    for status in (Status.FAILED, Status.ERROR):
        with pytest.raises(ValidationError, match="needs failure details"):
            make(status=status)


def test_passed_and_skipped_results_cannot_carry_a_failure() -> None:
    failure = FailureInfo(message="x", category=ErrorCategory.UNKNOWN)
    for status in (Status.PASSED, Status.SKIPPED):
        with pytest.raises(ValidationError, match="cannot carry a failure"):
            make(status=status, failure=failure)


def test_incomplete_may_explain_itself() -> None:
    result = make(
        status=Status.INCOMPLETE,
        failure=FailureInfo(message="judge returned no score", category=ErrorCategory.UNKNOWN),
    )

    assert result.status is Status.INCOMPLETE


def test_timestamps_must_be_aware_and_ordered() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make(started_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError, match="earlier than started_at"):
        make(finished_at=START - timedelta(seconds=1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "unknown"},
        {"module": "API Tests"},
        {"test_id": ""},
        {"unexpected_field": 1},
        {"metadata": {"path": Path("x")}},
        {"metadata": {"when": datetime.now(UTC)}},
        {"attempt": {"attempt": 4, "max_attempts": 3}},
    ],
)
def test_invalid_values_are_rejected_not_dropped(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        make(**overrides)


def test_non_finite_metric_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MetricResult(name="score", value=math.nan)


def test_results_are_immutable() -> None:
    with pytest.raises(ValidationError):
        make().status = Status.FAILED  # type: ignore[misc]
