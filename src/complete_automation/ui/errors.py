"""UI exceptions, all classified through the shared-core ``ErrorCategory``.

Failed UI assertions stay ``AssertionError`` (a test failure); a timeout while
acting on the page is a ``TIMEOUT``; a missing browser is a configuration problem.
"""

from __future__ import annotations

from complete_automation.core.exceptions import ErrorCategory, FrameworkError


class UiError(FrameworkError):
    """Base class for UI module errors."""


class UiAssertionError(UiError, AssertionError):
    """A UI expectation did not hold. The message carries the page URL and title."""

    category = ErrorCategory.ASSERTION


class UiTimeoutError(UiError, TimeoutError):
    """A page action or wait ran out of time (the element never became ready)."""

    category = ErrorCategory.TIMEOUT


class BrowserUnavailableError(UiError):
    """The configured browser engine cannot be started (usually not installed)."""

    category = ErrorCategory.CONFIGURATION
