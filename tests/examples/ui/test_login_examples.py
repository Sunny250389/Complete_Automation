"""Login examples against the local practice site (``tests/support/ui_site/login.html``).

They show the shape of a login test: a page object for the screens, test data read from ``data/ui/``
with the shared core loader, and assertions on what the user sees. The credentials are fake and exist
only on the practice site. For your own application, read real credentials from environment variables
(see docs/ui_automation.md) and never put them in a file in the repository.
"""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page

from complete_automation.core.datasets import DataLoader
from complete_automation.ui.assertions import (
    assert_text,
    assert_url,
    assert_validation_message,
    assert_visible,
)
from tests.support.ui_pages import AccountPage, LoginPage

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]

# Test data lives under data/ and is read through the loader, which refuses paths outside that folder.
LOGIN_CASES = DataLoader(Path(__file__).resolve().parents[3] / "data").load("ui/login_users.json")
REJECTED = [case for case in LOGIN_CASES if case["expect"] == "rejected"]
SIGNED_IN = next(case for case in LOGIN_CASES if case["expect"] == "signed_in")


def test_a_valid_login_reaches_the_account_page(ui_page: Page, ui_base_url: str) -> None:
    login = LoginPage(ui_page, ui_base_url)
    login.open()

    login.sign_in(SIGNED_IN["email"], SIGNED_IN["password"])

    assert_url(ui_page, re.compile(r"/account\.html$"))
    account = AccountPage(ui_page, ui_base_url)
    assert_text(account.signed_in.locator, f"Signed in as {SIGNED_IN['email']}", exact=True)


@pytest.mark.parametrize("case", REJECTED, ids=[case["case_id"] for case in REJECTED])
def test_a_rejected_login_stays_on_the_sign_in_page(ui_page: Page, ui_base_url: str, case: dict) -> None:
    login = LoginPage(ui_page, ui_base_url)
    login.open()

    login.sign_in(case["email"], case["password"])

    assert_visible(login.login_error.locator)
    assert_text(login.login_error.locator, "Email or password is incorrect", exact=True)
    assert_url(ui_page, re.compile(r"/login\.html$"))


def test_empty_fields_show_validation_messages(ui_page: Page, ui_base_url: str) -> None:
    login = LoginPage(ui_page, ui_base_url)
    login.open()

    login.sign_in("", "")

    assert_validation_message(login.email.error, "Email address is required")
    assert_validation_message(login.password.error, "Password is required")
    assert_url(ui_page, re.compile(r"/login\.html$"))
