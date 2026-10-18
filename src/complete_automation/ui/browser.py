"""Browser factory and lifecycle: one Playwright runtime and browser per session.

:func:`launch_browser` maps ``UiSettings`` to a launched engine; :class:`BrowserSession`
owns the Playwright runtime and the browser and releases both in a fixed order,
even when closing one of them fails. Contexts are cheap and isolated, so tests
get a fresh one each (see :meth:`BrowserSession.new_context`).

Importing this module requires the ``ui`` extra (``uv sync --extra ui``); it
starts no browser and opens no connection until :meth:`BrowserSession.start`.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Error as PlaywrightError, Playwright
from playwright.sync_api import expect, sync_playwright

from complete_automation.core.log import get_logger
from complete_automation.ui.errors import BrowserUnavailableError
from complete_automation.ui.settings import UiSettings

_LOG = get_logger("ui.browser")
_DEFAULT_EXPECT_TIMEOUT_MS = 5000


def launch_options(settings: UiSettings) -> dict[str, Any]:
    return {
        "headless": settings.headless,
        "args": list(settings.launch_args),
        "slow_mo": settings.slow_mo_ms,
    }


def context_options(settings: UiSettings, *, video_dir: Path | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "viewport": {"width": settings.viewport.width, "height": settings.viewport.height},
        "accept_downloads": False,
    }
    if settings.base_url:
        options["base_url"] = settings.base_url
    if settings.locale:
        options["locale"] = settings.locale.replace("_", "-")
    if settings.timezone_id:
        options["timezone_id"] = settings.timezone_id
    if video_dir is not None:
        options["record_video_dir"] = str(video_dir)
        options["record_video_size"] = options["viewport"]
    return options


def launch_browser(playwright: Playwright, settings: UiSettings) -> Browser:
    """Launch the configured engine.

    Raises:
        BrowserUnavailableError: the engine's binary is not installed. Other
            Playwright errors propagate unchanged.
    """
    engine = getattr(playwright, settings.browser)
    try:
        return engine.launch(**launch_options(settings))
    except PlaywrightError as error:
        if "Executable doesn't exist" in str(error) or "playwright install" in str(error):
            raise BrowserUnavailableError(
                f"{settings.browser} is not installed for Playwright; "
                f"run: playwright install {settings.browser}"
            ) from error
        raise


class BrowserSession:
    """A started Playwright runtime and one launched browser."""

    def __init__(self, settings: UiSettings, playwright: Playwright, browser: Browser) -> None:
        self.settings = settings
        self.playwright = playwright
        self.browser = browser
        self._closed = False

    @classmethod
    def start(cls, settings: UiSettings) -> BrowserSession:
        """Start Playwright and launch the browser; nothing stays running if that fails."""
        playwright = sync_playwright().start()
        try:
            playwright.selectors.set_test_id_attribute(settings.test_id_attribute)
            expect.set_options(timeout=settings.expect_timeout_ms)
            browser = launch_browser(playwright, settings)
        except BaseException:
            expect.set_options(timeout=_DEFAULT_EXPECT_TIMEOUT_MS)
            playwright.stop()
            raise
        _LOG.info("browser started", extra={"browser": settings.browser, "headless": settings.headless})
        return cls(settings, playwright, browser)

    def new_context(self, *, video_dir: Path | None = None) -> BrowserContext:
        """A fresh, isolated context with the configured timeouts applied."""
        if self._closed:
            raise RuntimeError("browser session is closed")
        context = self.browser.new_context(**context_options(self.settings, video_dir=video_dir))
        context.set_default_timeout(self.settings.action_timeout_ms)
        context.set_default_navigation_timeout(self.settings.navigation_timeout_ms)
        return context

    @property
    def version(self) -> str:
        return self.browser.version

    def close(self) -> None:
        """Close the browser, then stop Playwright. Idempotent; the first error is re-raised last."""
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for step in (self.browser.close, self.playwright.stop):
            try:
                step()
            except BaseException as error:  # release every resource, then re-raise
                _LOG.warning("closing browser resources failed: %s", type(error).__name__)
                first_error = first_error or error
        expect.set_options(timeout=_DEFAULT_EXPECT_TIMEOUT_MS)
        if first_error is not None:
            raise first_error

    def __enter__(self) -> BrowserSession:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
