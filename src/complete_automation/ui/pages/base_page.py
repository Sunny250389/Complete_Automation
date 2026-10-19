"""The base class for page objects.

A page object owns a page's locators and user-level actions; tests own the
intent and the assertions. Locators are created on demand (never cached), so
they always resolve against the current DOM, and Playwright's auto-waiting
decides when an action may proceed: there are no sleeps or retries here.

Prefer, in this order: ``by_role``, ``by_label``, ``by_test_id``, ``by_text``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from playwright.sync_api import Locator, Page, Response
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from complete_automation.ui.errors import UiTimeoutError

LoadState = Literal["load", "domcontentloaded", "networkidle"]
_ABSOLUTE_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


class BasePage:
    """Minimal, reusable page behaviour. Subclass it and set :attr:`path`."""

    path: ClassVar[str] = "/"
    """Location relative to the base URL; subclasses override it."""

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        self.page = page
        self.base_url = base_url

    # --- navigation -------------------------------------------------------------

    def open(self, *, wait_until: LoadState = "load") -> Response | None:
        """Navigate to :attr:`path` and return the main-frame response (if any)."""
        with self._translate(f"opening {self.path}"):
            return self.page.goto(self.absolute_url(self.path), wait_until=wait_until)

    def absolute_url(self, path: str = "") -> str:
        if _ABSOLUTE_URL.match(path):
            return path
        if not self.base_url:
            raise ValueError("base_url is required to open a relative path; configure modules.ui.base_url")
        return urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

    def wait_for_url(self, url: str | re.Pattern[str], *, timeout_ms: float | None = None) -> None:
        with self._translate(f"waiting for URL {url!r}"):
            self.page.wait_for_url(url, timeout=timeout_ms)

    def wait_for_load_state(self, state: LoadState = "load") -> None:
        with self._translate(f"waiting for load state {state}"):
            self.page.wait_for_load_state(state)

    @property
    def url(self) -> str:
        return self.page.url

    @property
    def title(self) -> str:
        return self.page.title()

    # --- locators ---------------------------------------------------------------

    def by_role(self, role: str, name: str | re.Pattern[str] | None = None, *, exact: bool = False) -> Locator:
        return self.page.get_by_role(role, name=name, exact=exact)  # type: ignore[arg-type]

    def by_label(self, text: str | re.Pattern[str], *, exact: bool = False) -> Locator:
        return self.page.get_by_label(text, exact=exact)

    def by_test_id(self, test_id: str) -> Locator:
        return self.page.get_by_test_id(test_id)

    def by_text(self, text: str | re.Pattern[str], *, exact: bool = False) -> Locator:
        return self.page.get_by_text(text, exact=exact)

    # --- actions (all wait for the element to be actionable) ----------------------

    def click(self, target: Locator) -> None:
        with self._translate(f"clicking {target}"):
            target.click()

    def fill(self, target: Locator, value: str) -> None:
        with self._translate(f"filling {target}"):
            target.fill(value)

    def select(self, target: Locator, *, label: str | None = None, value: str | None = None) -> None:
        if (label is None) == (value is None):
            raise ValueError("pass exactly one of label or value")
        with self._translate(f"selecting an option in {target}"):
            if label is not None:
                target.select_option(label=label)
            else:
                target.select_option(value=value)

    def set_checked(self, target: Locator, checked: bool = True) -> None:
        with self._translate(f"setting {target} checked={checked}"):
            target.set_checked(checked)

    def press(self, target: Locator, key: str) -> None:
        with self._translate(f"pressing {key} on {target}"):
            target.press(key)

    # --- reading ----------------------------------------------------------------

    def text_of(self, target: Locator) -> str:
        with self._translate(f"reading text of {target}"):
            return target.inner_text().strip()

    def attribute_of(self, target: Locator, name: str) -> str | None:
        with self._translate(f"reading attribute {name!r} of {target}"):
            return target.get_attribute(name)

    def value_of(self, target: Locator) -> str:
        with self._translate(f"reading value of {target}"):
            return target.input_value()

    def count_of(self, target: Locator) -> int:
        return target.count()

    def is_visible(self, target: Locator) -> bool:
        """Instantaneous check; to wait for visibility use ``assert_visible``."""
        return target.is_visible()

    def is_enabled(self, target: Locator) -> bool:
        with self._translate(f"checking enabled state of {target}"):
            return target.is_enabled()

    # --- diagnostics --------------------------------------------------------------

    def screenshot(self, *, full_page: bool = False) -> bytes:
        return self.page.screenshot(full_page=full_page)

    @contextmanager
    def accept_next_dialog(self) -> Iterator[list[str]]:
        """Accept the next alert/confirm/prompt; yields the list receiving its message."""
        messages: list[str] = []

        def handler(dialog: Any) -> None:
            messages.append(dialog.message)
            dialog.accept()

        self.page.once("dialog", handler)
        try:
            yield messages
        finally:
            self.page.remove_listener("dialog", handler)

    # --- internals ----------------------------------------------------------------

    @contextmanager
    def _translate(self, action: str) -> Iterator[None]:
        """Turn a Playwright timeout into :class:`UiTimeoutError` naming the action and page."""
        try:
            yield
        except PlaywrightTimeoutError as error:
            first_line = (str(error).strip().splitlines() or [""])[0]
            raise UiTimeoutError(f"timed out {action} on {self.page.url}: {first_line}") from error
