"""Per-test UI diagnostics, stored through the shared-core artifact store.

:class:`UiDiagnostics` watches one browser context for one test. When the test
ends it saves, according to ``modules.ui``: screenshots, the Playwright trace,
videos, a redacted browser console log and a ``diagnostics.json`` summary. Every
file becomes an ``ArtifactRef`` attached to that test's ``TestResult``, stored
under ``ui/<test name>-<hash of the test id>/`` so it can be traced back to the test.

Collection never raises: a failed screenshot or trace is logged and listed in
``metadata.ui.diagnostic_errors`` while the test keeps its original outcome.
Console text, URLs and failure text are redacted; screenshots, traces and
videos are not (see docs/ui_automation.md).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from playwright.sync_api import BrowserContext, ConsoleMessage, Page

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore, sanitize_name
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.redaction import Redactor
from complete_automation.ui.settings import UiSettings

MAX_CONSOLE_ENTRIES = 200
_CONSOLE_TEXT_LIMIT = 500
_FAILURE_TEXT_LIMIT = 4000
_SCREENSHOT_TIMEOUT_MS = 5000


def artifact_subdir(test_id: str, test_name: str) -> str:
    """``ui/<test name>-<8 hex of sha1(test id)>``: readable and unique per test and parameter set."""
    digest = hashlib.sha1(test_id.encode("utf-8")).hexdigest()[:8]
    return f"ui/{sanitize_name(test_name)[:80]}-{digest}"


class UiDiagnostics:
    def __init__(
        self,
        *,
        settings: UiSettings,
        store: ArtifactStore,
        recorder: ResultRecorder,
        redactor: Redactor,
        logger: logging.Logger,
        test_id: str,
        test_name: str,
        browser_version: str,
    ) -> None:
        self.settings = settings
        self.store = store
        self.recorder = recorder
        self.redactor = redactor
        self.logger = logger
        self.test_id = test_id
        self.subdir = artifact_subdir(test_id, test_name)
        self.browser_version = browser_version
        self.errors: list[str] = []
        self.video_dir: Path | None = None
        self._pages: list[Page] = []
        self._console: deque[dict[str, str]] = deque(maxlen=MAX_CONSOLE_ENTRIES)
        self._tracing = False

    # --- before the test -----------------------------------------------------------

    def prepare_video_dir(self) -> Path | None:
        """A scratch directory for Playwright's video files, or ``None`` when video is off."""
        if self.settings.video != "off":
            self.video_dir = Path(tempfile.mkdtemp(prefix="cauto-video-"))
        return self.video_dir

    def watch(self, context: BrowserContext) -> None:
        """Start listening to the context (console, page errors, new pages) and tracing if enabled."""
        context.on("page", self._watch_page)
        for page in context.pages:
            self._watch_page(page)
        if self.settings.trace != "off":
            try:
                context.tracing.start(screenshots=True, snapshots=True, sources=False)
                self._tracing = True
            except Exception as error:
                self._problem("starting the trace", error)

    def _watch_page(self, page: Page) -> None:
        self._pages.append(page)
        page.on("console", self._on_console)
        page.on("pageerror", lambda error: self._add_console("pageerror", str(error)))

    def _on_console(self, message: ConsoleMessage) -> None:
        self._add_console(message.type, message.text)

    def _add_console(self, kind: str, text: str) -> None:
        self._console.append({"type": kind, "text": self.redactor.redact_text(text)[:_CONSOLE_TEXT_LIMIT]})

    # --- after the test ---------------------------------------------------------------

    def before_close(self, context: BrowserContext, *, failed: bool, failure_text: str | None) -> None:
        """Save what needs a live context: screenshots, trace, console log, summary. Never raises."""
        try:
            if self._wants(self.settings.screenshot, failed):
                self._screenshots()
            self._finish_trace(context, failed)
            if failed or self.settings.screenshot == "always":
                self._save_console()
            self._record_summary(failed, failure_text)
        except Exception as error:  # last resort: diagnostics must not change the test outcome
            self._problem("collecting diagnostics", error)

    def after_close(self, *, failed: bool) -> None:
        """Save videos (finalised when the context closed) and remove the scratch directory."""
        try:
            if self.video_dir is not None and self._wants(self.settings.video, failed):
                self._videos()
        except Exception as error:
            self._problem("saving videos", error)
        finally:
            self._remove_video_dir()
            if self.errors:  # videos can fail after the summary was recorded
                self.recorder.add_metadata(ui=self._ui_metadata())

    # --- steps ---------------------------------------------------------------------------

    @staticmethod
    def _wants(policy: str, failed: bool) -> bool:
        return policy == "always" or (policy == "on_failure" and failed)

    def _screenshots(self) -> None:
        open_pages = [page for page in self._pages if not page.is_closed()]
        for index, page in enumerate(open_pages[: self.settings.max_screenshots_per_test], start=1):
            try:
                data = page.screenshot(timeout=_SCREENSHOT_TIMEOUT_MS)
                self._attach(
                    self.store.write_bytes(
                        f"screenshot-{index}.png",
                        data,
                        subdir=self.subdir,
                        media_type="image/png",
                        metadata=self._artifact_metadata("screenshot", page_url=self._safe_url(page)),
                    )
                )
            except Exception as error:
                self._problem(f"taking screenshot {index}", error)

    def _finish_trace(self, context: BrowserContext, failed: bool) -> None:
        if not self._tracing:
            return
        self._tracing = False
        try:
            if not self._wants(self.settings.trace, failed):
                context.tracing.stop()
                return
            with tempfile.TemporaryDirectory(prefix="cauto-trace-") as scratch:
                target = Path(scratch) / "trace.zip"
                context.tracing.stop(path=str(target))
                self._attach(
                    self.store.write_bytes(
                        "trace.zip",
                        target.read_bytes(),
                        subdir=self.subdir,
                        media_type="application/zip",
                        metadata=self._artifact_metadata("trace"),
                    )
                )
        except Exception as error:
            self._problem("saving the trace", error)

    def _save_console(self) -> None:
        if not self._console:
            return
        try:
            self._attach(
                self.store.write_json(
                    "console.json",
                    list(self._console),
                    subdir=self.subdir,
                    metadata=self._artifact_metadata("console", entries=len(self._console)),
                )
            )
        except Exception as error:
            self._problem("saving the console log", error)

    def _videos(self) -> None:
        for index, page in enumerate(self._pages, start=1):
            try:
                video = page.video
                if video is None:
                    continue
                data = Path(video.path()).read_bytes()
                self._attach(
                    self.store.write_bytes(
                        f"video-{index}.webm",
                        data,
                        subdir=self.subdir,
                        media_type="video/webm",
                        metadata=self._artifact_metadata("video"),
                    )
                )
            except Exception as error:
                self._problem(f"saving video {index}", error)

    def _record_summary(self, failed: bool, failure_text: str | None) -> None:
        ui = self._ui_metadata()
        self.recorder.add_metadata(ui=ui)
        if not failed and self.settings.screenshot != "always":
            return
        summary: dict[str, Any] = {
            "test_id": self.test_id,
            "outcome": "failed" if failed else "passed",
            "ui": ui,
            "settings": self.redactor.redact(self.settings.model_dump(mode="json")),
        }
        if failure_text:
            summary["failure"] = self.redactor.redact_text(failure_text)[:_FAILURE_TEXT_LIMIT]
        try:
            self._attach(
                self.store.write_json(
                    "diagnostics.json",
                    summary,
                    subdir=self.subdir,
                    metadata=self._artifact_metadata("diagnostics"),
                )
            )
        except Exception as error:
            self._problem("saving diagnostics.json", error)

    # --- helpers ---------------------------------------------------------------------------

    def _ui_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "browser": self.settings.browser,
            "browser_version": self.browser_version,
            "headless": self.settings.headless,
            "viewport": f"{self.settings.viewport.width}x{self.settings.viewport.height}",
            "base_url": self.redactor.redact_text(self.settings.base_url or ""),
            "page_urls": [self._safe_url(page) for page in self._pages],
        }
        if self.errors:
            metadata["diagnostic_errors"] = list(self.errors)
        return metadata

    def _safe_url(self, page: Page) -> str:
        return self.redactor.redact_text(page.url)

    def _artifact_metadata(self, kind: str, **extra: Any) -> dict[str, Any]:
        return {"test_id": self.test_id, "kind": kind, "browser": self.settings.browser, **extra}

    def _attach(self, ref: ArtifactRef) -> None:
        self.recorder.attach(ref)

    def _problem(self, action: str, error: BaseException) -> None:
        text = self.redactor.redact_text(f"{action} failed: {type(error).__name__}: {error}")
        text = (text.strip().splitlines() or [""])[0][:300]
        self.errors.append(text)
        self.logger.warning("UI diagnostics: %s", text)

    def _remove_video_dir(self) -> None:
        if self.video_dir is None:
            return
        shutil.rmtree(self.video_dir, ignore_errors=True)
        self.video_dir = None
