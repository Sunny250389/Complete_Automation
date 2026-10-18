"""UI automation module: browser lifecycle, page objects, assertions, diagnostics and fixtures.

Builds on ``complete_automation.core`` for configuration, logging, redaction,
results, reporting and artifacts, and on Playwright (the ``ui`` extra:
``uv sync --extra ui``, then ``playwright install``). Importing this package
needs neither Playwright nor a browser; ``ui.browser``, ``ui.pages``,
``ui.assertions``, ``ui.diagnostics`` and ``ui.pytest_plugin`` import Playwright.
"""

from complete_automation.ui.errors import (
    BrowserUnavailableError,
    UiAssertionError,
    UiError,
    UiTimeoutError,
)
from complete_automation.ui.settings import UiSettings

__all__ = [
    "BrowserUnavailableError",
    "UiAssertionError",
    "UiError",
    "UiSettings",
    "UiTimeoutError",
]
