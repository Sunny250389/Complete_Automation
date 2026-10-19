"""UI assertions: Playwright web-first ``expect`` checks with page context in the failure.

Each helper retries until the expectation holds or the timeout
(``modules.ui.expect_timeout_seconds`` unless ``timeout_ms`` is given) runs out,
then raises :class:`UiAssertionError`, an ``AssertionError`` whose message adds the
page URL and title to Playwright's own explanation. The original error is chained.
Nothing is retried beyond Playwright's built-in polling, and nothing is swallowed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from playwright.sync_api import Locator, Page, expect

from complete_automation.core.redaction import DEFAULT_REDACTOR
from complete_automation.ui.errors import UiAssertionError

TextMatch = str | re.Pattern[str]
_MESSAGE_LIMIT = 2000


def _page_of(subject: Page | Locator) -> Page:
    return subject if isinstance(subject, Page) else subject.page


def _context(page: Page) -> str:
    try:
        title = page.title()
    except Exception:  # the page may already be closed; the URL alone still helps
        title = "<unavailable>"
    return f"page: {page.url} (title: {title!r})"


def _verify(subject: Page | Locator, description: str, check: Callable[[], None]) -> None:
    try:
        check()
    except AssertionError as error:
        detail = str(error).strip()[:_MESSAGE_LIMIT]
        message = f"{description}\n{detail}\n{_context(_page_of(subject))}"
        raise UiAssertionError(DEFAULT_REDACTOR.redact_text(message)) from error


# --- elements -------------------------------------------------------------------


def assert_visible(target: Locator, *, timeout_ms: float | None = None) -> None:
    _verify(target, f"expected {target} to be visible", lambda: expect(target).to_be_visible(timeout=timeout_ms))


def assert_hidden(target: Locator, *, timeout_ms: float | None = None) -> None:
    """Passes when the element is not visible or not in the DOM at all."""
    _verify(target, f"expected {target} to be hidden", lambda: expect(target).to_be_hidden(timeout=timeout_ms))


def assert_text(
    target: Locator, expected: TextMatch, *, exact: bool = False, timeout_ms: float | None = None
) -> None:
    """Text content of the element: a substring match, or the full text when ``exact``."""
    _verify(
        target,
        f"expected {target} to have text {expected!r}",
        lambda: expect(target).to_have_text(expected, timeout=timeout_ms)
        if exact
        else expect(target).to_contain_text(expected, timeout=timeout_ms),
    )


def assert_count(target: Locator, expected: int, *, timeout_ms: float | None = None) -> None:
    _verify(
        target,
        f"expected {target} to match {expected} element(s)",
        lambda: expect(target).to_have_count(expected, timeout=timeout_ms),
    )


def assert_enabled(target: Locator, *, timeout_ms: float | None = None) -> None:
    _verify(target, f"expected {target} to be enabled", lambda: expect(target).to_be_enabled(timeout=timeout_ms))


def assert_disabled(target: Locator, *, timeout_ms: float | None = None) -> None:
    _verify(target, f"expected {target} to be disabled", lambda: expect(target).to_be_disabled(timeout=timeout_ms))


def assert_attribute(
    target: Locator, name: str, expected: TextMatch, *, timeout_ms: float | None = None
) -> None:
    _verify(
        target,
        f"expected {target} to have attribute {name}={expected!r}",
        lambda: expect(target).to_have_attribute(name, expected, timeout=timeout_ms),
    )


def assert_value(target: Locator, expected: TextMatch, *, timeout_ms: float | None = None) -> None:
    _verify(
        target,
        f"expected {target} to have value {expected!r}",
        lambda: expect(target).to_have_value(expected, timeout=timeout_ms),
    )


# --- form controls ----------------------------------------------------------------


def assert_checked(target: Locator, checked: bool = True, *, timeout_ms: float | None = None) -> None:
    """Checkbox or radio button state."""
    state = "checked" if checked else "unchecked"
    _verify(
        target,
        f"expected {target} to be {state}",
        lambda: expect(target).to_be_checked(checked=checked, timeout=timeout_ms),
    )


def assert_selected_option(
    dropdown: Locator, expected: TextMatch, *, timeout_ms: float | None = None
) -> None:
    """The value of the selected option of a ``<select>``."""
    _verify(
        dropdown,
        f"expected {dropdown} to have selected value {expected!r}",
        lambda: expect(dropdown).to_have_value(expected, timeout=timeout_ms),
    )


def assert_validation_message(
    message: Locator, expected: TextMatch, *, timeout_ms: float | None = None
) -> None:
    """The visible validation message of a form field (its error element) shows ``expected``."""
    _verify(
        message,
        f"expected validation message {expected!r} in {message}",
        lambda: (
            expect(message).to_be_visible(timeout=timeout_ms),
            expect(message).to_contain_text(expected, timeout=timeout_ms),
        ),
    )


# --- page ---------------------------------------------------------------------------


def assert_url(page: Page, expected: TextMatch, *, timeout_ms: float | None = None) -> None:
    """The page URL equals ``expected`` (string) or matches it (regular expression)."""
    _verify(page, f"expected URL {expected!r}", lambda: expect(page).to_have_url(expected, timeout=timeout_ms))


def assert_title(page: Page, expected: TextMatch, *, timeout_ms: float | None = None) -> None:
    _verify(page, f"expected title {expected!r}", lambda: expect(page).to_have_title(expected, timeout=timeout_ms))
