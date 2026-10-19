"""Small components shared by page objects: a labelled form field and a status notice."""

from __future__ import annotations

from playwright.sync_api import Locator, Page


class FormField:
    """A form control found by its exact accessible label, plus its error message element.

    The error is located by ``error_test_id``; pages whose errors follow another
    convention can pass any ``Locator`` as ``error``.
    """

    def __init__(
        self,
        page: Page,
        label: str,
        *,
        error: Locator | None = None,
        error_test_id: str | None = None,
    ) -> None:
        self.label = label
        self.control = page.get_by_label(label, exact=True)
        if error is not None:
            self.error = error
        elif error_test_id is not None:
            self.error = page.get_by_test_id(error_test_id)
        else:
            raise ValueError("pass error or error_test_id")

    def fill(self, value: str) -> None:
        self.control.fill(value)

    def value(self) -> str:
        return self.control.input_value()

    def __repr__(self) -> str:
        return f"FormField({self.label!r})"


class Notice:
    """A status or alert region identified by test id."""

    def __init__(self, page: Page, test_id: str) -> None:
        self.locator = page.get_by_test_id(test_id)

    def text(self) -> str:
        return self.locator.inner_text().strip()

    def is_visible(self) -> bool:
        return self.locator.is_visible()
