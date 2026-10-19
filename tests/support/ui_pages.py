"""Page objects for the local practice site (``tests/support/ui_site``).

They show the intended shape of a page object: locators and user actions live
here, test intent and assertions stay in the test.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page

from complete_automation.ui.pages import BasePage, FormField, Notice


class HomePage(BasePage):
    path = "/"

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        self.heading = self.by_role("heading", "Practice Home")
        self.features = self.by_test_id("feature")
        self.registration_link = self.by_role("link", "Registration form")
        self.load_button = self.by_role("button", "Load details")
        self.details = self.by_test_id("details")

    def load_details(self) -> Locator:
        self.click(self.load_button)
        return self.details

    def open_registration(self) -> RegistrationPage:
        self.click(self.registration_link)
        return RegistrationPage(self.page, self.base_url)


class RegistrationPage(BasePage):
    path = "/form.html"

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        self.heading = self.by_role("heading", "Create an account")
        self.name = FormField(page, "Full name", error_test_id="name-error")
        self.email = FormField(page, "Email address", error_test_id="email-error")
        self.country = self.by_label("Country", exact=True)
        self.free_plan = self.by_role("radio", "Free")
        self.pro_plan = self.by_role("radio", "Pro")
        self.newsletter = self.by_label("Send me the newsletter", exact=False)
        self.terms = self.by_label("I accept the terms")
        self.terms_error = self.by_test_id("terms-error")
        self.submit_button = self.by_role("button", "Create account")
        self.confirmation = Notice(page, "confirmation")

    def submit(self) -> None:
        self.click(self.submit_button)

    def register(
        self, name: str, email: str, *, country: str | None = None, plan: str = "Free", accept_terms: bool = True
    ) -> None:
        self.name.fill(name)
        self.email.fill(email)
        if country:
            self.select(self.country, label=country)
        self.click(self.pro_plan if plan == "Pro" else self.free_plan)
        self.set_checked(self.terms, accept_terms)
        self.submit()
