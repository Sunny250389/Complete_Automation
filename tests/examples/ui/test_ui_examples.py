"""Runnable UI examples against the local practice site.

These show how a test uses the fixtures (``ui_page``, ``ui_base_url``), page objects
and assertions. They are framework examples, not tests of a real application; to
test your own application see docs/ui_automation.md.
"""

import re

import pytest
from playwright.sync_api import Page

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.ui.assertions import (
    assert_attribute,
    assert_checked,
    assert_count,
    assert_disabled,
    assert_enabled,
    assert_hidden,
    assert_selected_option,
    assert_text,
    assert_title,
    assert_url,
    assert_validation_message,
    assert_visible,
)
from tests.support.ui_pages import HomePage, RegistrationPage

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]


def test_navigation_and_title(ui_page: Page, ui_base_url: str) -> None:
    home = HomePage(ui_page, ui_base_url)
    home.open()

    assert_title(ui_page, "Practice Site - Home")
    assert_url(ui_page, re.compile(r"/$"))

    registration = home.open_registration()

    assert_url(ui_page, re.compile(r"/form\.html$"))
    assert_title(ui_page, "Practice Site - Registration")
    assert_visible(registration.heading)


def test_locating_elements_and_waiting_for_late_content(ui_page: Page, ui_base_url: str) -> None:
    home = HomePage(ui_page, ui_base_url)
    home.open()

    assert_visible(home.heading)
    assert_count(home.features, 3)
    assert_hidden(home.details)  # not shown until the button is used

    details = home.load_details()  # appears after ~300 ms: auto-waiting, no sleep

    assert_visible(details)
    assert_text(details, "Details loaded", exact=True)


def test_form_submission(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    form.register("Ada Lovelace", "ada@example.test", country="Germany", plan="Pro")

    assert_visible(form.confirmation.locator)
    assert_text(form.confirmation.locator, "Welcome, Ada Lovelace! Plan: pro, country: Germany", exact=True)
    assert_url(ui_page, re.compile(r"/form\.html#submitted$"))


def test_validation_messages(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    form.name.fill("Ada")
    form.email.fill("not-an-email")
    form.submit()

    assert_validation_message(form.email.error, "Enter a valid email address")
    assert_validation_message(form.terms_error, "You must accept the terms")
    assert_text(form.name.error, "", exact=True)  # the name was valid: no message
    assert_hidden(form.confirmation.locator)


def test_dropdown_radio_and_checkbox(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    form.select(form.country, label="India")
    assert_selected_option(form.country, "in")

    assert_checked(form.free_plan)  # the default plan
    assert_disabled(form.newsletter)  # Pro-only option
    form.click(form.pro_plan)
    assert_checked(form.pro_plan)
    assert_checked(form.free_plan, checked=False)
    assert_enabled(form.newsletter)

    form.set_checked(form.terms)
    assert_checked(form.terms)
    assert_attribute(form.terms, "type", "checkbox")


def test_attaching_a_screenshot_to_the_result(
    ui_page: Page,
    ui_base_url: str,
    artifact_store: LocalArtifactStore,
    result_recorder: ResultRecorder,
) -> None:
    """On-demand evidence: store a screenshot with the shared artifact store and attach it to this test's result.

    Failures are captured automatically; see tests/integration/ui/test_failure_diagnostics.py.
    """
    home = HomePage(ui_page, ui_base_url)
    home.open()

    ref = artifact_store.write_bytes(
        "home.png", home.screenshot(), subdir="ui/examples", media_type="image/png"
    )
    result_recorder.attach(ref)

    assert ref.size_bytes > 0
    assert artifact_store.resolve(ref).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
