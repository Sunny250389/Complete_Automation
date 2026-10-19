"""BrowserSession and the browser factory with fake Playwright objects: no browser is launched."""

from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError

from complete_automation.ui import browser as browser_module
from complete_automation.ui.browser import BrowserSession, launch_browser
from complete_automation.ui.errors import BrowserUnavailableError
from complete_automation.ui.settings import UiSettings

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]


class FakeSelectors:
    def __init__(self) -> None:
        self.attribute: str | None = None

    def set_test_id_attribute(self, name: str) -> None:
        self.attribute = name


class FakeContext:
    def __init__(self) -> None:
        self.timeouts: dict[str, float] = {}

    def set_default_timeout(self, ms: float) -> None:
        self.timeouts["action"] = ms

    def set_default_navigation_timeout(self, ms: float) -> None:
        self.timeouts["navigation"] = ms


class FakeBrowser:
    version = "1.2.3"

    def __init__(self, events: list[str], fail_close: bool = False) -> None:
        self.events, self.fail_close = events, fail_close
        self.context = FakeContext()

    def new_context(self, **options: Any) -> FakeContext:
        return self.context

    def close(self) -> None:
        self.events.append("browser.close")
        if self.fail_close:
            raise RuntimeError("browser close failed")


class FakeEngine:
    def __init__(self, browser: FakeBrowser | None = None, error: Exception | None = None) -> None:
        self.browser, self.error = browser, error
        self.launch_args: dict[str, Any] = {}

    def launch(self, **options: Any) -> FakeBrowser | None:
        self.launch_args = options
        if self.error:
            raise self.error
        return self.browser


class FakePlaywright:
    def __init__(self, events: list[str], **engines: FakeEngine) -> None:
        self.events = events
        self.selectors = FakeSelectors()
        self.__dict__.update(engines)

    def stop(self) -> None:
        self.events.append("playwright.stop")


@pytest.fixture
def events() -> list[str]:
    return []


def patch_runtime(monkeypatch: pytest.MonkeyPatch, playwright: FakePlaywright) -> None:
    class Starter:
        def start(self) -> FakePlaywright:
            return playwright

    monkeypatch.setattr(browser_module, "sync_playwright", lambda: Starter())


def test_launch_uses_the_configured_engine_and_options(events: list[str]) -> None:
    firefox, chromium = FakeEngine(FakeBrowser(events)), FakeEngine(FakeBrowser(events))
    playwright = FakePlaywright(events, firefox=firefox, chromium=chromium)

    launch_browser(playwright, UiSettings(browser="firefox", headless=False))  # type: ignore[arg-type]

    assert firefox.launch_args["headless"] is False and chromium.launch_args == {}


def test_a_missing_binary_becomes_an_actionable_error(events: list[str]) -> None:
    missing = PlaywrightError("BrowserType.launch: Executable doesn't exist at C:\\x\\chrome.exe")
    playwright = FakePlaywright(events, webkit=FakeEngine(error=missing))

    with pytest.raises(BrowserUnavailableError, match="playwright install webkit") as caught:
        launch_browser(playwright, UiSettings(browser="webkit"))  # type: ignore[arg-type]

    assert caught.value.__cause__ is missing


def test_other_launch_errors_are_not_rewritten(events: list[str]) -> None:
    broken = PlaywrightError("Target page, context or browser has been closed")
    playwright = FakePlaywright(events, chromium=FakeEngine(error=broken))

    with pytest.raises(PlaywrightError, match="has been closed"):
        launch_browser(playwright, UiSettings())  # type: ignore[arg-type]


def test_start_applies_settings_and_close_releases_in_order(
    monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> None:
    playwright = FakePlaywright(events, chromium=FakeEngine(FakeBrowser(events)))
    patch_runtime(monkeypatch, playwright)
    settings = UiSettings(test_id_attribute="data-qa", action_timeout_seconds=2, navigation_timeout_seconds=7)

    session = BrowserSession.start(settings)
    context = session.new_context()
    session.close()
    session.close()  # idempotent

    assert playwright.selectors.attribute == "data-qa"
    assert context.timeouts == {"action": 2000, "navigation": 7000}  # type: ignore[attr-defined]
    assert events == ["browser.close", "playwright.stop"]  # exactly once, browser first
    assert session.version == "1.2.3"
    with pytest.raises(RuntimeError, match="closed"):
        session.new_context()


def test_failed_launch_stops_playwright(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    missing = PlaywrightError("Executable doesn't exist")
    playwright = FakePlaywright(events, chromium=FakeEngine(error=missing))
    patch_runtime(monkeypatch, playwright)

    with pytest.raises(BrowserUnavailableError):
        BrowserSession.start(UiSettings())

    assert events == ["playwright.stop"]


def test_close_stops_playwright_even_when_the_browser_fails_to_close(
    monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> None:
    playwright = FakePlaywright(events, chromium=FakeEngine(FakeBrowser(events, fail_close=True)))
    patch_runtime(monkeypatch, playwright)
    session = BrowserSession.start(UiSettings())

    with pytest.raises(RuntimeError, match="browser close failed"):
        session.close()

    assert events == ["browser.close", "playwright.stop"]  # nothing leaked, error not swallowed


def test_context_manager_closes_on_exception(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    playwright = FakePlaywright(events, chromium=FakeEngine(FakeBrowser(events)))
    patch_runtime(monkeypatch, playwright)

    with pytest.raises(ValueError, match="test body"):
        with BrowserSession.start(UiSettings()):
            raise ValueError("test body")

    assert events == ["browser.close", "playwright.stop"]
