"""UiDiagnostics with fake Playwright objects: naming, association, redaction and failure isolation."""

import logging
from pathlib import Path
from typing import Any

import pytest

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.exceptions import ArtifactError
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.redaction import Redactor
from complete_automation.ui.diagnostics import MAX_CONSOLE_ENTRIES, UiDiagnostics, artifact_subdir
from complete_automation.ui.settings import UiSettings

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]

PNG = b"\x89PNG\r\n\x1a\n-fake-"


class FakeTracing:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.started, self.stopped_with = fail, False, "not stopped"

    def start(self, **options: Any) -> None:
        self.started = True

    def stop(self, path: str | None = None) -> None:
        if self.fail:
            raise RuntimeError("trace writer crashed")
        self.stopped_with = path
        if path:
            Path(path).write_bytes(b"PK-fake-trace")


class FakePage:
    def __init__(self, url: str = "about:blank", fail_screenshot: bool = False) -> None:
        self.url, self.fail_screenshot, self.handlers = url, fail_screenshot, {}
        self.video = None

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def is_closed(self) -> bool:
        return False

    def screenshot(self, **options: Any) -> bytes:
        if self.fail_screenshot:
            raise RuntimeError("cannot capture token=abc123secret")
        return PNG


class FakeContext:
    def __init__(self, pages: list[FakePage], tracing: FakeTracing | None = None) -> None:
        self.pages, self.tracing, self.handlers = pages, tracing or FakeTracing(), {}

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler


def make(
    tmp_path: Path, test_id: str = "tests/ui/test_a.py::test_x[1]", **settings: Any
) -> tuple[UiDiagnostics, LocalArtifactStore, ResultRecorder]:
    store = LocalArtifactStore(tmp_path / "store")
    recorder = ResultRecorder()
    diagnostics = UiDiagnostics(
        settings=UiSettings(**settings),
        store=store,
        recorder=recorder,
        redactor=Redactor(),
        logger=logging.getLogger("test-ui-diagnostics"),
        test_id=test_id,
        test_name=test_id.rsplit("::", 1)[-1],
        browser_version="1.0",
    )
    return diagnostics, store, recorder


def test_artifact_folder_is_readable_unique_and_safe() -> None:
    first = artifact_subdir("tests/ui/test_a.py::test_x[1]", "test_x[1]")
    second = artifact_subdir("tests/ui/test_a.py::test_x[2]", "test_x[2]")
    other_file = artifact_subdir("tests/ui/test_b.py::test_x[1]", "test_x[1]")

    assert len({first, second, other_file}) == 3  # same name in another file or parameter set differs
    assert first.startswith("ui/test_x_1_-") and "[" not in first
    assert artifact_subdir("a::" + "n" * 500, "n" * 500).count("/") == 1  # one safe folder level


def test_failure_stores_screenshot_with_the_test_identity(tmp_path: Path) -> None:
    diagnostics, store, recorder = make(tmp_path)
    page = FakePage("http://app.test/cart?session_token=hunter2&page=2")
    context = FakeContext([page])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=True, failure_text="AssertionError: nope")  # type: ignore[arg-type]
    diagnostics.after_close(failed=True)

    kinds = {a.metadata["kind"]: a for a in recorder.artifacts}
    assert set(kinds) == {"screenshot", "diagnostics"}
    shot = kinds["screenshot"]
    assert shot.metadata["test_id"] == "tests/ui/test_a.py::test_x[1]"
    assert shot.path.startswith(diagnostics.subdir + "/") and shot.media_type == "image/png"
    assert "hunter2" not in str(shot.metadata["page_url"]) and "page=2" in str(shot.metadata["page_url"])
    assert store.resolve(shot).read_bytes() == PNG
    assert recorder.metadata["ui"]["browser"] == "chromium"  # type: ignore[index]


def test_passing_test_stores_nothing_by_default(tmp_path: Path) -> None:
    diagnostics, store, recorder = make(tmp_path)
    context = FakeContext([FakePage()])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=False, failure_text=None)  # type: ignore[arg-type]
    diagnostics.after_close(failed=False)

    assert recorder.artifacts == [] and not store.root.exists()
    assert "diagnostic_errors" not in recorder.metadata["ui"]  # type: ignore[operator]


def test_never_policy_stores_no_screenshot_even_on_failure(tmp_path: Path) -> None:
    diagnostics, _, recorder = make(tmp_path, screenshot="never")
    context = FakeContext([FakePage()])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=True, failure_text=None)  # type: ignore[arg-type]

    assert {a.metadata["kind"] for a in recorder.artifacts} == {"diagnostics"}


def test_screenshot_count_is_capped(tmp_path: Path) -> None:
    diagnostics, _, recorder = make(tmp_path, max_screenshots_per_test=2)
    context = FakeContext([FakePage(), FakePage(), FakePage()])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=True, failure_text=None)  # type: ignore[arg-type]

    assert sorted(a.name for a in recorder.artifacts if a.metadata["kind"] == "screenshot") == [
        "screenshot-1.png",
        "screenshot-2.png",
    ]


def test_a_failing_screenshot_is_recorded_not_raised_and_other_artifacts_still_saved(tmp_path: Path) -> None:
    diagnostics, _, recorder = make(tmp_path)
    context = FakeContext([FakePage(fail_screenshot=True), FakePage()])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=True, failure_text="AssertionError: original")  # type: ignore[arg-type]
    diagnostics.after_close(failed=True)  # must not raise

    assert [a.name for a in recorder.artifacts if a.metadata["kind"] == "screenshot"] == ["screenshot-2.png"]
    errors = recorder.metadata["ui"]["diagnostic_errors"]  # type: ignore[index]
    assert len(errors) == 1 and "screenshot 1" in errors[0]
    assert "abc123secret" not in errors[0]  # redacted


def test_a_failing_artifact_store_does_not_raise(tmp_path: Path) -> None:
    diagnostics, store, recorder = make(tmp_path)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise ArtifactError("disk full")

    store.write_bytes = broken  # type: ignore[method-assign]
    store.write_json = broken  # type: ignore[method-assign]
    context = FakeContext([FakePage()])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=True, failure_text="x")  # type: ignore[arg-type]
    diagnostics.after_close(failed=True)

    assert recorder.artifacts == []
    assert any("disk full" in e for e in recorder.metadata["ui"]["diagnostic_errors"])  # type: ignore[index]


def test_trace_is_kept_on_failure_and_discarded_otherwise(tmp_path: Path) -> None:
    failing, _, recorder = make(tmp_path, trace="on_failure", screenshot="never")
    context = FakeContext([FakePage()])
    failing.watch(context)  # type: ignore[arg-type]
    failing.before_close(context, failed=True, failure_text=None)  # type: ignore[arg-type]
    assert "trace" in {a.metadata["kind"] for a in recorder.artifacts}

    passing, _, recorder = make(tmp_path / "pass", trace="on_failure", screenshot="never")
    context = FakeContext([FakePage()])
    passing.watch(context)  # type: ignore[arg-type]
    passing.before_close(context, failed=False, failure_text=None)  # type: ignore[arg-type]
    assert recorder.artifacts == [] and context.tracing.stopped_with is None


def test_trace_is_not_started_when_off(tmp_path: Path) -> None:
    diagnostics, _, _ = make(tmp_path)
    context = FakeContext([FakePage()])

    diagnostics.watch(context)  # type: ignore[arg-type]

    assert context.tracing.started is False


def test_a_trace_failure_is_isolated(tmp_path: Path) -> None:
    diagnostics, _, recorder = make(tmp_path, trace="always")
    context = FakeContext([FakePage()], FakeTracing(fail=True))
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(context, failed=True, failure_text=None)  # type: ignore[arg-type]

    assert {a.metadata["kind"] for a in recorder.artifacts} == {"screenshot", "diagnostics"}
    assert any("trace" in e for e in recorder.metadata["ui"]["diagnostic_errors"])  # type: ignore[index]


def test_console_messages_are_redacted_bounded_and_stored(tmp_path: Path) -> None:
    diagnostics, store, recorder = make(tmp_path)
    page = FakePage()
    context = FakeContext([page])
    diagnostics.watch(context)  # type: ignore[arg-type]

    class Message:
        type = "error"
        text = "login failed password=hunter2"

    page.handlers["console"](Message())
    for number in range(MAX_CONSOLE_ENTRIES + 20):
        page.handlers["pageerror"](f"error {number}")
    diagnostics.before_close(context, failed=True, failure_text=None)  # type: ignore[arg-type]

    console = next(a for a in recorder.artifacts if a.metadata["kind"] == "console")
    text = store.resolve(console).read_text("utf-8")
    assert "hunter2" not in text and console.metadata["entries"] == MAX_CONSOLE_ENTRIES
    assert f"error {MAX_CONSOLE_ENTRIES + 19}" in text and "login failed" not in text  # oldest dropped


def test_new_pages_opened_during_the_test_are_watched(tmp_path: Path) -> None:
    diagnostics, _, recorder = make(tmp_path)
    context = FakeContext([])
    diagnostics.watch(context)  # type: ignore[arg-type]

    context.handlers["page"](FakePage("http://app.test/popup"))
    diagnostics.before_close(context, failed=True, failure_text=None)  # type: ignore[arg-type]

    assert recorder.metadata["ui"]["page_urls"] == ["http://app.test/popup"]  # type: ignore[index]
    assert any(a.metadata["kind"] == "screenshot" for a in recorder.artifacts)


def test_failure_text_in_the_summary_is_redacted_and_truncated(tmp_path: Path) -> None:
    diagnostics, store, recorder = make(tmp_path)
    context = FakeContext([FakePage()])
    diagnostics.watch(context)  # type: ignore[arg-type]

    diagnostics.before_close(  # type: ignore[arg-type]
        context, failed=True, failure_text="Authorization: Bearer abc.def.ghi\n" + "x" * 10000
    )

    ref = next(a for a in recorder.artifacts if a.metadata["kind"] == "diagnostics")
    summary = store.resolve(ref).read_text("utf-8")
    assert "abc.def.ghi" not in summary and len(summary) < 8000
