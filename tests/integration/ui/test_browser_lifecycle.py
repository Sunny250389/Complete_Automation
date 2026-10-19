"""Browser, context and page lifecycle with a real browser.

A second Playwright runtime cannot start next to the session's one (the sync API forbids it),
so tests that start and stop their own ``BrowserSession`` do so in a subprocess.
"""

import json
import subprocess
import sys
import textwrap

import pytest
from playwright.sync_api import BrowserContext, Page

from complete_automation.ui.browser import BrowserSession

pytestmark = [pytest.mark.ui, pytest.mark.integration, pytest.mark.cauto_module("ui")]


def _run(script: str) -> dict:  # type: ignore[type-arg]
    completed = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_each_test_gets_its_own_context_and_page(ui_context: BrowserContext, ui_session: BrowserSession) -> None:
    first = ui_context.new_page()
    second = ui_context.new_page()

    assert first is not second
    assert len(ui_context.pages) == 2
    assert ui_context in ui_session.browser.contexts
    assert len(ui_session.browser.contexts) == 1  # no leftover contexts from earlier tests


def test_state_set_in_one_test_is_invisible_to_the_next_a(ui_page: Page, ui_base_url: str) -> None:
    ui_page.goto(ui_base_url + "/")
    ui_page.evaluate("localStorage.setItem('left-behind', 'yes')")
    ui_page.context.add_cookies([{"name": "session", "value": "x", "url": ui_base_url}])

    assert ui_page.evaluate("localStorage.getItem('left-behind')") == "yes"


def test_state_set_in_one_test_is_invisible_to_the_next_b(ui_page: Page, ui_base_url: str) -> None:
    ui_page.goto(ui_base_url + "/")

    assert ui_page.evaluate("localStorage.getItem('left-behind')") is None
    assert ui_page.context.cookies() == []


def test_context_uses_the_configured_viewport_and_base_url(ui_page: Page, ui_settings, ui_base_url: str) -> None:  # type: ignore[no-untyped-def]
    ui_page.goto("/form.html")  # relative: resolved against the context's base_url

    assert ui_page.url == f"{ui_base_url}/form.html"
    assert ui_page.evaluate("[innerWidth, innerHeight]") == [
        ui_settings.viewport.width,
        ui_settings.viewport.height,
    ]


def test_session_applies_settings_and_releases_everything_on_close() -> None:
    result = _run(
        """
        import json
        from complete_automation.ui.browser import BrowserSession
        from complete_automation.ui.settings import UiSettings

        settings = UiSettings(
            viewport={"width": 800, "height": 600}, locale="de-DE", timezone_id="Europe/Berlin",
            test_id_attribute="data-qa", action_timeout_seconds=0.3, navigation_timeout_seconds=2,
        )
        session = BrowserSession.start(settings)
        context = session.new_context()
        page = context.new_page()
        page.set_content('<div data-qa="hello">Hi</div><a href="/x">x</a>')
        seen = {
            "size": page.evaluate("[innerWidth, innerHeight]"),
            "language": page.evaluate("navigator.language"),
            "zone": page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone"),
            "test_id": page.get_by_test_id("hello").inner_text(),
        }
        try:
            page.get_by_role("button", name="missing").click()
        except Exception as error:
            seen["action_timeout"] = type(error).__name__
        browser = session.browser
        session.close()
        session.close()  # idempotent
        seen["browser_connected_after_close"] = browser.is_connected()
        try:
            session.new_context()
        except RuntimeError as error:
            seen["new_context_after_close"] = str(error)
        print(json.dumps(seen))
        """
    )

    assert result["size"] == [800, 600]
    assert result["language"] == "de-DE" and result["zone"] == "Europe/Berlin"
    assert result["test_id"] == "Hi"
    assert result["action_timeout"] == "TimeoutError"  # the configured 0.3 s, not Playwright's 30 s
    assert result["browser_connected_after_close"] is False
    assert result["new_context_after_close"] == "browser session is closed"
