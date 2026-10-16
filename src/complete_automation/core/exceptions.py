"""Framework exception hierarchy and error classification.

Retry guidance:
- ``ConfigurationError``, ``TestDataError``: never retry; fix the input.
- ``ArtifactError``, ``ReportingError``: infrastructure; surface immediately.
- Domain modules decide which of their own transient errors are retryable and
  pass them to :mod:`complete_automation.core.retry` explicitly.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Why a test did not pass. Shared by exceptions and test results."""

    ASSERTION = "assertion"
    """A test check failed: the system under test misbehaved."""
    THRESHOLD = "threshold"
    """An evaluation metric fell below its acceptance threshold."""
    INFRASTRUCTURE = "infrastructure"
    """A dependency of the test failed: service, network, file system, storage."""
    CONFIGURATION = "configuration"
    TEST_DATA = "test_data"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FrameworkError(Exception):
    """Base class for errors raised by the framework.

    Domain modules may subclass it. ``category`` lets reporting classify the
    failure without importing the domain module.
    """

    category: ErrorCategory = ErrorCategory.UNKNOWN


class ConfigurationError(FrameworkError):
    category = ErrorCategory.CONFIGURATION


class TestDataError(FrameworkError):
    __test__ = False  # not a pytest test class
    category = ErrorCategory.TEST_DATA


class ArtifactError(FrameworkError):
    category = ErrorCategory.INFRASTRUCTURE


class ReportingError(FrameworkError):
    category = ErrorCategory.INFRASTRUCTURE


class RetryExhaustedError(FrameworkError):
    """Every allowed attempt raised a retryable error.

    The last underlying error is chained as ``__cause__`` and kept in ``last_error``.
    """

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(
            f"gave up after {attempts} attempt(s); last error: "
            f"{type(last_error).__name__}: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error

    @property
    def category(self) -> ErrorCategory:  # type: ignore[override]
        return classify_exception(self.last_error)


def classify_exception(error: BaseException) -> ErrorCategory:
    """Map an exception to an :class:`ErrorCategory`.

    Framework errors carry their own category. ``AssertionError`` is a test
    failure; timeouts and connection or OS errors are infrastructure problems;
    anything else is ``UNKNOWN`` (usually a defect in the test or the framework).
    """
    if isinstance(error, FrameworkError):
        return error.category
    if isinstance(error, AssertionError):
        return ErrorCategory.ASSERTION
    if isinstance(error, TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(error, (ConnectionError, OSError)):
        return ErrorCategory.INFRASTRUCTURE
    return ErrorCategory.UNKNOWN
