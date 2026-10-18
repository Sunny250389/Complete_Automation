"""BasePage helpers and UI assertions against a real browser and the local practice site."""

import re

import pytest
from playwright.sync_api import Page

from complete_automation.core.exceptions import ErrorCategory, classify_exception
from complete_automation.ui import UiAssertionError, UiTimeoutError
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
    assert_value,
    assert_visible,
)
from complete_automation.ui.pages import BasePage
from tests.support.ui_pages import HomePage, RegistrationPage

pytestmark = [pytest.mark.ui, pytest.mark.integration, pytest.mark.cauto_module("ui")]

SHORT_MS = 300


# --- BasePage ---------------------------------------------------------------------


def test_open_navigates_relative_to_the_base_url(ui_page: Page, ui_base_url: str) -> None:
    response = RegistrationPage(ui_page, ui_base_url).open()

    assert response is not None and response.status == 200
    assert ui_page.url == f"{ui_base_url}/form.html"


def test_relative_path_without_base_url_is_a_clear_error(ui_page: Page) -> None:
    with pytest.raises(ValueError, match="modules.ui.base_url"):
        BasePage(ui_page).open()


def test_absolute_urls_pass_through(ui_page: Page) -> None:
    assert BasePage(ui_page, "http://a.test").absolute_url("https://b.test/x") == "https://b.test/x"
    assert BasePage(ui_page, "http://a.test/").absolute_url("/x/y") == "http://a.test/x/y"


def test_readers_return_page_state(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()
    form.name.fill("Grace")

    assert form.title == "Practice Site - Registration"
    assert form.url.endswith("/form.html")
    assert form.value_of(form.name.control) == "Grace"
    assert form.text_of(form.heading) == "Create an account"
    assert form.attribute_of(form.terms, "type") == "checkbox"
    assert form.count_of(form.by_role("radio")) == 2
    assert form.is_visible(form.heading) and not form.is_visible(form.confirmation.locator)
    assert form.is_enabled(form.free_plan) and not form.is_enabled(form.newsletter)


def test_select_requires_exactly_one_criterion(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    with pytest.raises(ValueError, match="exactly one"):
        form.select(form.country)
    with pytest.raises(ValueError, match="exactly one"):
        form.select(form.country, label="India", value="in")
    form.select(form.country, value="de")
    assert form.value_of(form.country) == "de"


def test_a_missing_element_is_a_timeout_error_naming_the_action_and_page(
    ui_page: Page, ui_base_url: str
) -> None:
    ui_page.set_default_timeout(SHORT_MS)
    home = HomePage(ui_page, ui_base_url)
    home.open()

    with pytest.raises(UiTimeoutError, match=r"clicking .*on http://127\.0\.0\.1:\d+/") as caught:
        home.click(home.by_role("button", "No such button"))

    assert classify_exception(caught.value) is ErrorCategory.TIMEOUT
    assert caught.value.__cause__ is not None  # the Playwright error is chained, not hidden


def test_wait_for_url_follows_navigation_and_times_out_otherwise(ui_page: Page, ui_base_url: str) -> None:
    home = HomePage(ui_page, ui_base_url)
    home.open()
    home.click(home.registration_link)

    home.wait_for_url(re.compile(r"/form\.html$"))
    with pytest.raises(UiTimeoutError, match="waiting for URL"):
        home.wait_for_url("**/never.html", timeout_ms=SHORT_MS)


def test_accept_next_dialog_captures_the_message(ui_page: Page) -> None:
    ui_page.set_content('<button onclick="alert(\'saved\')">Save</button>')
    page = BasePage(ui_page)

    with page.accept_next_dialog() as messages:
        page.click(page.by_role("button", "Save"))

    assert messages == ["saved"]


# --- assertions that pass -------------------------------------------------------------


def test_assertions_pass_on_matching_state(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()
    form.register("Ada", "ada@example.test", country="Canada")

    assert_visible(form.confirmation.locator)
    assert_hidden(form.submit_button)
    assert_text(form.confirmation.locator, re.compile(r"^Welcome, Ada!"))
    assert_url(ui_page, re.compile(r"#submitted$"))
    assert_title(ui_page, "Practice Site - Registration")
    assert_count(form.by_role("heading"), 1)


def test_assertions_cover_form_state(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    assert_enabled(form.free_plan)
    assert_disabled(form.newsletter)
    assert_checked(form.terms, checked=False)
    assert_value(form.name.control, "")
    assert_selected_option(form.country, "")
    assert_attribute(form.name.control, "id", "name")
    form.submit()
    assert_validation_message(form.name.error, "Full name is required")


# --- assertions that fail: message quality --------------------------------------------


def test_failed_assertion_is_an_assertion_error_with_page_context(ui_page: Page, ui_base_url: str) -> None:
    home = HomePage(ui_page, ui_base_url)
    home.open()

    with pytest.raises(UiAssertionError) as caught:
        assert_text(home.heading, "Something else", exact=True, timeout_ms=SHORT_MS)

    error = caught.value
    assert isinstance(error, AssertionError)  # pytest reports it as a failure, not an error
    assert classify_exception(error) is ErrorCategory.ASSERTION
    message = str(error)
    assert "Something else" in message
    assert f"page: {ui_base_url}/" in message and "Practice Site - Home" in message
    assert isinstance(error.__cause__, AssertionError)


@pytest.mark.parametrize(
    "check",
    [
        lambda page, form: assert_visible(form.confirmation.locator, timeout_ms=SHORT_MS),
        lambda page, form: assert_hidden(form.heading, timeout_ms=SHORT_MS),
        lambda page, form: assert_count(form.by_role("radio"), 5, timeout_ms=SHORT_MS),
        lambda page, form: assert_enabled(form.newsletter, timeout_ms=SHORT_MS),
        lambda page, form: assert_disabled(form.free_plan, timeout_ms=SHORT_MS),
        lambda page, form: assert_checked(form.terms, timeout_ms=SHORT_MS),
        lambda page, form: assert_selected_option(form.country, "de", timeout_ms=SHORT_MS),
        lambda page, form: assert_attribute(form.terms, "type", "radio", timeout_ms=SHORT_MS),
        lambda page, form: assert_validation_message(form.name.error, "anything", timeout_ms=SHORT_MS),
        lambda page, form: assert_url(page, "http://example.invalid/", timeout_ms=SHORT_MS),
        lambda page, form: assert_title(page, "Wrong title", timeout_ms=SHORT_MS),
    ],
    ids=[
        "visible", "hidden", "count", "enabled", "disabled", "checked",
        "selected-option", "attribute", "validation-message", "url", "title",
    ],
)
def test_each_assertion_fails_when_the_expectation_does_not_hold(
    ui_page: Page, ui_base_url: str, check
) -> None:  # type: ignore[no-untyped-def]
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    with pytest.raises(UiAssertionError, match="page: "):
        check(ui_page, form)


def test_assertion_failure_text_is_redacted(ui_page: Page) -> None:
    ui_page.set_content('<p id="p">ok</p>')

    with pytest.raises(UiAssertionError) as caught:
        assert_text(ui_page.locator("#p"), "token=super-secret-value", exact=True, timeout_ms=SHORT_MS)

    assert "super-secret-value" not in str(caught.value)
