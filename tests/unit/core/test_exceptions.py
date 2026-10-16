import pytest

from complete_automation.core.exceptions import (
    ArtifactError,
    ConfigurationError,
    ErrorCategory,
    FrameworkError,
    ReportingError,
    RetryExhaustedError,
    TestDataError,
    classify_exception,
)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ConfigurationError("x"), ErrorCategory.CONFIGURATION),
        (TestDataError("x"), ErrorCategory.TEST_DATA),
        (ArtifactError("x"), ErrorCategory.INFRASTRUCTURE),
        (ReportingError("x"), ErrorCategory.INFRASTRUCTURE),
        (FrameworkError("x"), ErrorCategory.UNKNOWN),
        (AssertionError("x"), ErrorCategory.ASSERTION),
        (TimeoutError("x"), ErrorCategory.TIMEOUT),
        (ConnectionRefusedError("x"), ErrorCategory.INFRASTRUCTURE),
        (FileNotFoundError("x"), ErrorCategory.INFRASTRUCTURE),
        (ValueError("x"), ErrorCategory.UNKNOWN),
        (KeyError("x"), ErrorCategory.UNKNOWN),
    ],
)
def test_classification(error: BaseException, category: ErrorCategory) -> None:
    assert classify_exception(error) is category


def test_domain_modules_can_declare_their_own_category() -> None:
    class ServiceUnavailable(FrameworkError):
        category = ErrorCategory.INFRASTRUCTURE

    assert classify_exception(ServiceUnavailable("503")) is ErrorCategory.INFRASTRUCTURE
    assert isinstance(ServiceUnavailable("503"), FrameworkError)


def test_retry_exhaustion_is_classified_by_its_last_error() -> None:
    assert RetryExhaustedError(3, TimeoutError()).category is ErrorCategory.TIMEOUT
    assert RetryExhaustedError(3, ConnectionError()).category is ErrorCategory.INFRASTRUCTURE


def test_chaining_preserves_the_original_error() -> None:
    with pytest.raises(ConfigurationError) as raised:
        try:
            {}["missing"]
        except KeyError as error:
            raise ConfigurationError("bad config") from error

    assert isinstance(raised.value.__cause__, KeyError)


def test_framework_errors_share_one_base() -> None:
    for kind in (ConfigurationError, TestDataError, ArtifactError, ReportingError, RetryExhaustedError):
        assert issubclass(kind, FrameworkError)
