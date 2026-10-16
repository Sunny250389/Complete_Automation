"""Shared pytest fixtures and the result-reporting hook.

Enable it from a top-level ``conftest.py``::

    pytest_plugins = ["complete_automation.core.pytest_plugin"]

Command-line options:

``--cauto-env NAME``         environment to load (beats ``CAUTO_ENVIRONMENT``)
``--cauto-config-dir PATH``  configuration directory
``--cauto-report``           write a JSON report (also ``reporting.json_enabled``)

Fixtures (no autouse, no network, nothing created until first use):

=====================  ========  ==============================================
``framework_settings``  session  validated :class:`FrameworkSettings`
``run_context``         session  :class:`RunContext` for this run
``framework_logger``    session  ``complete_automation.tests`` logger
``artifact_store``      session  :class:`LocalArtifactStore` under ``<artifacts.root>/<run_id>``
``test_context``        function :class:`TestContext`, made current for the test
``result_recorder``     function metrics, artifacts and metadata for this test's result
=====================  ========  ==============================================

When reporting is enabled, every test produces exactly one :class:`TestResult`
(published after its teardown) and the report is written at session end.
Mark a test with ``@pytest.mark.cauto_module("api")`` to set its module;
otherwise the folder under ``tests/`` is used.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from complete_automation.core.artifacts import ArtifactRef, LocalArtifactStore
from complete_automation.core.config import FrameworkSettings, load_settings
from complete_automation.core.context import (
    RunContext,
    TestContext,
    activate_run_context,
    deactivate_run_context,
    new_correlation_id,
    use_test_context,
)
from complete_automation.core.exceptions import ConfigurationError, ErrorCategory, classify_exception
from complete_automation.core.log import configure_logging, get_logger, reset_logging
from complete_automation.core.reporting import JsonReporter
from complete_automation.core.results import FailureInfo, MetricResult, Status, TestResult

_MESSAGE_LIMIT = 500
_DETAILS_LIMIT = 8000
_NON_TAG_MARKERS = frozenset(
    {"parametrize", "usefixtures", "filterwarnings", "skip", "skipif", "xfail", "cauto_module"}
)
_JSON = TypeAdapter(dict[str, JsonValue])


class ResultRecorder:
    """Collects domain detail for the current test's :class:`TestResult`.

    Values are validated when added, so a bad metric fails the test that added
    it rather than the report at the end of the run.
    """

    def __init__(self) -> None:
        self.metrics: list[MetricResult] = []
        self.artifacts: list[ArtifactRef] = []
        self.metadata: dict[str, JsonValue] = {}
        self.incomplete_reason: str | None = None

    def add_metric(
        self,
        name: str,
        value: float | int | bool | str,
        *,
        threshold: float | None = None,
        passed: bool | None = None,
        unit: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> MetricResult:
        metric = MetricResult(
            name=name,
            value=value,
            threshold=threshold,
            passed=passed,
            unit=unit,
            metadata=metadata or {},
        )
        self.metrics.append(metric)
        return metric

    def attach(self, ref: ArtifactRef) -> None:
        if not isinstance(ref, ArtifactRef):
            raise TypeError(f"expected ArtifactRef, got {type(ref).__name__}")
        self.artifacts.append(ref)

    def add_metadata(self, **values: Any) -> None:
        try:
            self.metadata.update(_JSON.validate_python(values))
        except ValidationError:
            raise TypeError("metadata values must be JSON-compatible") from None

    def mark_incomplete(self, reason: str) -> None:
        """Record that the test ran but reached no verdict (status ``incomplete``)."""
        self.incomplete_reason = reason


@dataclass
class _State:
    settings: FrameworkSettings
    run: RunContext
    reporter: JsonReporter | None
    run_token: Any
    report_path: Path | None = None


_STATE = pytest.StashKey[_State]()
_PHASES = pytest.StashKey[dict[str, tuple[pytest.TestReport, pytest.CallInfo[Any]]]]()
_RECORDER = pytest.StashKey[ResultRecorder]()
_TEST_CONTEXT = pytest.StashKey[TestContext]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("complete_automation", "Complete_Automation shared core")
    group.addoption("--cauto-env", default=None, help="environment to load")
    group.addoption("--cauto-config-dir", default=None, help="configuration directory")
    group.addoption(
        "--cauto-report", action="store_true", default=False, help="write a JSON results report"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "cauto_module(name): testing module reported for this test (api, ui, ...)"
    )
    try:
        settings = load_settings(
            config_dir=config.getoption("--cauto-config-dir"),
            environment=config.getoption("--cauto-env"),
        )
    except ConfigurationError as error:
        raise pytest.UsageError(f"complete_automation configuration: {error}") from error

    configure_logging(settings.logging, redactor=settings.redactor())
    run = RunContext.start(settings.environment, run_id=settings.execution.run_id)
    reporter = None
    if config.getoption("--cauto-report") or settings.reporting.json_enabled:
        reporter = JsonReporter(_report_path(settings, run), redactor=settings.redactor())
        reporter.start_run(run)
    config.stash[_STATE] = _State(settings, run, reporter, activate_run_context(run))


def pytest_unconfigure(config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    if state is None:
        return
    deactivate_run_context(state.run_token)
    reset_logging()


def pytest_sessionfinish(session: pytest.Session) -> None:
    state = session.config.stash.get(_STATE, None)
    if state is None:
        return
    state.run = state.run.finish()
    if state.reporter is not None:
        state.report_path = state.reporter.finalize(state.run)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    if state is not None and state.report_path is not None:
        terminalreporter.write_line(f"complete_automation JSON report: {state.report_path}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    state = item.config.stash.get(_STATE, None)
    if state is not None and state.reporter is not None:
        phases = item.stash.setdefault(_PHASES, {})
        phases[call.when] = (report, call)
        if call.when == "teardown":
            state.reporter.publish(_build_result(item, phases, state))
    return report


@pytest.fixture(scope="session")
def framework_settings(pytestconfig: pytest.Config) -> FrameworkSettings:
    return pytestconfig.stash[_STATE].settings


@pytest.fixture(scope="session")
def run_context(pytestconfig: pytest.Config) -> RunContext:
    return pytestconfig.stash[_STATE].run


@pytest.fixture(scope="session")
def framework_logger() -> logging.Logger:
    return get_logger("tests")


@pytest.fixture(scope="session")
def artifact_store(framework_settings: FrameworkSettings, run_context: RunContext) -> LocalArtifactStore:
    return LocalArtifactStore.for_run(framework_settings.artifacts.root, run_context.run_id)


@pytest.fixture
def test_context(request: pytest.FixtureRequest, run_context: RunContext) -> Iterator[TestContext]:
    context = TestContext(
        run_id=run_context.run_id,
        test_id=request.node.nodeid,
        name=request.node.name,
        module=_module_of(request.node),
    )
    request.node.stash[_TEST_CONTEXT] = context
    with use_test_context(context):
        yield context


@pytest.fixture
def result_recorder(request: pytest.FixtureRequest) -> ResultRecorder:
    recorder = ResultRecorder()
    request.node.stash[_RECORDER] = recorder
    return recorder


def _report_path(settings: FrameworkSettings, run: RunContext) -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    name = f"results-{worker}.json" if worker else "results.json"
    return settings.reporting.output_dir / run.run_id / name


def _module_of(item: pytest.Item) -> str:
    marker = item.get_closest_marker("cauto_module")
    if marker is not None and marker.args:
        name = str(marker.args[0])
    else:
        try:
            parts = item.path.relative_to(item.config.rootpath).parts
        except ValueError:
            parts = item.path.parts
        name = parts[1] if len(parts) > 2 and parts[0] == "tests" else "default"
    cleaned = re.sub(r"[^a-z0-9_.-]+", "_", name.lower()).strip("_.-")
    return cleaned or "default"


def _timestamp(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, UTC)


def _failure(report: pytest.TestReport, call: pytest.CallInfo[Any], state: _State) -> FailureInfo:
    redactor = state.settings.redactor()
    error = call.excinfo.value if call.excinfo is not None else None
    if error is not None:
        first_line = (str(error).strip().splitlines() or [""])[0]
        message = f"{type(error).__name__}: {first_line}" if first_line else type(error).__name__
        category = classify_exception(error)
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
    else:  # e.g. strict XPASS: pytest reports a failure without an exception
        message = report.longreprtext.strip().splitlines()[0] if report.longreprtext else report.outcome
        category = ErrorCategory.ASSERTION
        exception_type = None
    details = report.longreprtext[:_DETAILS_LIMIT] if report.longreprtext else None
    return FailureInfo(
        message=redactor.redact_text(message)[:_MESSAGE_LIMIT],
        category=category,
        exception_type=exception_type,
        details=redactor.redact_text(details) if details else None,
    )


def _skip_reason(report: pytest.TestReport) -> str:
    if hasattr(report, "wasxfail"):
        return f"xfail: {report.wasxfail}" if report.wasxfail else "xfail"
    if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        return str(report.longrepr[2]).removeprefix("Skipped: ")
    return "skipped"


def _build_result(
    item: pytest.Item,
    phases: dict[str, tuple[pytest.TestReport, pytest.CallInfo[Any]]],
    state: _State,
) -> TestResult:
    recorder = item.stash.get(_RECORDER, None) or ResultRecorder()
    context = item.stash.get(_TEST_CONTEXT, None)
    metadata: dict[str, JsonValue] = dict(recorder.metadata)
    failure: FailureInfo | None = None

    setup_report, setup_call = phases["setup"]
    teardown_report, teardown_call = phases["teardown"]
    if setup_report.failed:
        status, failure = Status.ERROR, _failure(setup_report, setup_call, state)
    elif setup_report.skipped:
        status = Status.SKIPPED
        metadata["skip_reason"] = _skip_reason(setup_report)
    elif "call" not in phases:  # e.g. --setup-only: the test body never ran
        status = Status.SKIPPED
        metadata["skip_reason"] = "test body not run"
    else:
        call_report, call_call = phases["call"]
        if call_report.passed:
            status = Status.PASSED
            if hasattr(call_report, "wasxfail"):
                metadata["xpass"] = True
        elif call_report.skipped:
            status = Status.SKIPPED
            metadata["skip_reason"] = _skip_reason(call_report)
        else:
            failure = _failure(call_report, call_call, state)
            if failure.category is ErrorCategory.ASSERTION and any(
                metric.passed is False for metric in recorder.metrics
            ):
                failure = failure.model_copy(update={"category": ErrorCategory.THRESHOLD})
            status = (
                Status.FAILED
                if failure.category in (ErrorCategory.ASSERTION, ErrorCategory.THRESHOLD)
                else Status.ERROR
            )

    if teardown_report.failed and status in (Status.PASSED, Status.SKIPPED):
        status, failure = Status.ERROR, _failure(teardown_report, teardown_call, state)
        metadata.pop("skip_reason", None)
    elif teardown_report.failed:
        metadata["teardown_failed"] = True

    if status is Status.PASSED and recorder.incomplete_reason is not None:
        status = Status.INCOMPLETE
        failure = FailureInfo(
            message=state.settings.redactor().redact_text(recorder.incomplete_reason)[:_MESSAGE_LIMIT],
            category=ErrorCategory.UNKNOWN,
        )

    return TestResult(
        run_id=state.run.run_id,
        test_id=item.nodeid,
        name=item.name,
        module=_module_of(item),
        status=status,
        started_at=_timestamp(setup_call.start),
        finished_at=_timestamp(max(teardown_call.stop, setup_call.start)),
        correlation_id=context.correlation_id if context else new_correlation_id(),
        failure=failure,
        metrics=tuple(recorder.metrics),
        artifacts=tuple(recorder.artifacts),
        metadata=metadata,
        tags=tuple(sorted({m.name for m in item.iter_markers()} - _NON_TAG_MARKERS)),
    )
