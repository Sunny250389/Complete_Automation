"""Failure diagnostics inside real pytest runs: what is stored, where, for which test, and with what safeguards.

Each scenario runs a small pytest session in a subprocess (a second Playwright runtime cannot
share this process) and inspects the shared-core JSON report and artifact store it produced.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ui, pytest.mark.integration, pytest.mark.cauto_module("ui")]

CONFTEST = (
    'pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.ui.pytest_plugin"]\n'
)
PAGE = "<h1>Checkout</h1><p id='state'>ready</p>"
SECRET = "not-a-real-secret-value"

TESTS = f"""
import pytest

pytestmark = pytest.mark.cauto_module("ui")

def test_passes(ui_page):
    ui_page.set_content({PAGE!r})
    assert ui_page.locator("#state").inner_text() == "ready"

def test_fails_on_an_assertion(ui_page):
    ui_page.set_content({PAGE!r})
    ui_page.evaluate("console.error('call failed: token={SECRET}')")
    assert ui_page.locator("#state").inner_text() == "done"

@pytest.mark.parametrize("n", [1, 2])
def test_parametrized_failure(ui_page, n):
    ui_page.set_content({PAGE!r})
    raise AssertionError("case %d failed" % n)

def test_second_page_is_captured_too(ui_context):
    ui_context.new_page().set_content("<h1>one</h1>")
    ui_context.new_page().set_content("<h1>two</h1>")
    assert False, "boom"
"""


@pytest.fixture
def project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    for name in list(os.environ):
        keep = name in ("CAUTO_MODULES__UI__BROWSER", "CAUTO_MODULES__UI__HEADLESS")
        if name.startswith("CAUTO_") and not keep:
            monkeypatch.delenv(name)

    def make(ui_yaml: str = "") -> Path:
        config = pytester.path / "config"
        (config / "environments").mkdir(parents=True, exist_ok=True)
        body = "reporting:\n  json_enabled: true\n"
        if ui_yaml:
            body += "modules:\n  ui:\n" + "".join(f"    {line}\n" for line in ui_yaml.strip().splitlines())
        (config / "environments" / "dev.yaml").write_text(body, encoding="utf-8")
        monkeypatch.setenv("CAUTO_CONFIG_DIR", str(config))
        pytester.makeconftest(CONFTEST)
        return pytester.path

    return make


def _run(pytester: pytest.Pytester, source: str = TESTS) -> dict[str, dict]:  # type: ignore[type-arg]
    pytester.makepyfile(source)
    pytester.runpytest_subprocess("-p", "no:cacheprovider")
    report = next((pytester.path / "reports").glob("*/results.json"))
    return {r["name"]: r for r in json.loads(report.read_text(encoding="utf-8"))["results"]}


def _store_root(pytester: pytest.Pytester) -> Path:
    return next((pytester.path / "artifacts").iterdir())


def _kinds(result: dict) -> dict[str, dict]:  # type: ignore[type-arg]
    return {a["metadata"]["kind"]: a for a in result["artifacts"]}


def test_defaults_capture_a_screenshot_only_for_failures(
    pytester: pytest.Pytester, project  # type: ignore[no-untyped-def]
) -> None:
    project()

    results = _run(pytester)

    assert results["test_passes"]["status"] == "passed"
    assert results["test_passes"]["artifacts"] == []  # nothing stored when a test passes
    failed = results["test_fails_on_an_assertion"]
    assert failed["status"] == "failed"
    kinds = _kinds(failed)
    assert set(kinds) == {"screenshot", "console", "diagnostics"}  # trace and video are off by default
    screenshot = _store_root(pytester) / kinds["screenshot"]["path"]
    assert screenshot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert kinds["screenshot"]["media_type"] == "image/png"


def test_artifacts_are_stored_under_the_failing_test_and_never_shared(
    pytester: pytest.Pytester, project  # type: ignore[no-untyped-def]
) -> None:
    project()

    results = _run(pytester)

    seen: dict[str, str] = {}
    for name in ("test_fails_on_an_assertion", "test_parametrized_failure[1]", "test_parametrized_failure[2]"):
        result = results[name]
        for artifact in result["artifacts"]:
            assert artifact["metadata"]["test_id"] == result["test_id"]
            assert artifact["path"].startswith("ui/")
            assert artifact["path"] not in seen, "two tests share one artifact"
            seen[artifact["path"]] = name
        folders = {a["path"].rsplit("/", 1)[0] for a in result["artifacts"]}
        assert len(folders) == 1 and name.split("[")[0] in folders.pop()
    assert len({p.rsplit("/", 1)[0] for p in seen}) == 3  # parameter sets get separate folders


def test_diagnostics_json_describes_the_run_and_the_failure(
    pytester: pytest.Pytester, project  # type: ignore[no-untyped-def]
) -> None:
    project("viewport: {width: 640, height: 480}")

    results = _run(pytester)

    failed = results["test_fails_on_an_assertion"]
    summary = json.loads((_store_root(pytester) / _kinds(failed)["diagnostics"]["path"]).read_text("utf-8"))
    assert summary["test_id"] == failed["test_id"] and summary["outcome"] == "failed"
    assert summary["ui"]["viewport"] == "640x480" and isinstance(summary["ui"]["headless"], bool)
    assert summary["ui"]["page_urls"] == ["about:blank"]
    assert "assert" in summary["failure"] and "done" in summary["failure"]
    assert failed["metadata"]["ui"]["browser"] == summary["ui"]["browser"]
    assert failed["failure"]["category"] == "assertion"


def test_console_log_is_stored_redacted(pytester: pytest.Pytester, project) -> None:  # type: ignore[no-untyped-def]
    project()

    results = _run(pytester)

    path = _store_root(pytester) / _kinds(results["test_fails_on_an_assertion"])["console"]["path"]
    text = path.read_text("utf-8")
    assert "call failed" in text and SECRET not in text and "token=***" in text


def test_every_open_page_is_captured_up_to_the_limit(
    pytester: pytest.Pytester, project  # type: ignore[no-untyped-def]
) -> None:
    project("max_screenshots_per_test: 2")

    result = _run(pytester)["test_second_page_is_captured_too"]

    names = sorted(a["name"] for a in result["artifacts"] if a["metadata"]["kind"] == "screenshot")
    assert names == ["screenshot-1.png", "screenshot-2.png"]


def test_trace_and_video_are_opt_in_and_stored(pytester: pytest.Pytester, project) -> None:  # type: ignore[no-untyped-def]
    project("trace: on_failure\nvideo: on_failure\nscreenshot: never")

    results = _run(pytester)

    failed = _kinds(results["test_fails_on_an_assertion"])
    assert {"trace", "video", "console", "diagnostics"} <= set(failed) and "screenshot" not in failed
    trace = _store_root(pytester) / failed["trace"]["path"]
    assert trace.read_bytes()[:2] == b"PK"  # a zip archive
    video = _store_root(pytester) / failed["video"]["path"]
    assert video.stat().st_size > 0 and failed["video"]["media_type"] == "video/webm"
    assert results["test_passes"]["artifacts"] == []  # on_failure discards them for passing tests


def test_always_policies_also_store_for_passing_tests(
    pytester: pytest.Pytester, project  # type: ignore[no-untyped-def]
) -> None:
    project("screenshot: always\ntrace: always")

    results = _run(pytester)

    assert {"screenshot", "trace", "diagnostics"} <= set(_kinds(results["test_passes"]))


def test_scratch_files_are_removed(pytester: pytest.Pytester, project) -> None:  # type: ignore[no-untyped-def]
    project("video: always")
    before = {p.name for p in Path(os.environ.get("TEMP", "/tmp")).glob("cauto-video-*")}

    _run(pytester)

    after = {p.name for p in Path(os.environ.get("TEMP", "/tmp")).glob("cauto-video-*")}
    assert after <= before


def test_a_failing_screenshot_does_not_mask_the_test_failure(
    pytester: pytest.Pytester, project  # type: ignore[no-untyped-def]
) -> None:
    project()
    pytester.makeconftest(
        CONFTEST
        + """
import pytest
from playwright.sync_api import Page

@pytest.fixture(autouse=True)
def _broken_screenshots(monkeypatch):
    def broken(self, **kwargs):
        raise RuntimeError("screenshot backend exploded")
    monkeypatch.setattr(Page, "screenshot", broken)
"""
    )

    results = _run(pytester)

    failed = results["test_fails_on_an_assertion"]
    assert failed["status"] == "failed" and failed["failure"]["category"] == "assertion"
    assert "done" in failed["failure"]["details"]  # the original assertion, not the diagnostic error
    assert "screenshot" not in _kinds(failed)
    errors = failed["metadata"]["ui"]["diagnostic_errors"]
    assert any("screenshot backend exploded" in e for e in errors)


def test_a_missing_browser_is_a_clear_configuration_error(
    pytester: pytest.Pytester, project, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    project()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(pytester.path / "no-browsers"))

    results = _run(pytester, "def test_x(ui_page):\n    pass\n")

    result = results["test_x"]
    assert result["status"] == "error" and result["failure"]["category"] == "configuration"
    assert "playwright install" in result["failure"]["message"]
