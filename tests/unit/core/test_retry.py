import random

import pytest

from complete_automation.core.config import RetrySettings
from complete_automation.core.exceptions import RetryExhaustedError
from complete_automation.core.retry import RetryPolicy, call_with_retry, retrying

pytestmark = pytest.mark.cauto_module("core")


class Flaky:
    """Raises the given errors in order, then returns "ok"."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture
def sleeps() -> list[float]:
    return []


def policy(**overrides) -> RetryPolicy:
    values = dict(retry_on=(ConnectionError,), max_attempts=3, delay_seconds=0.5, backoff_multiplier=2.0)
    values.update(overrides)
    return RetryPolicy(**values)


def test_success_on_first_attempt_does_not_sleep(sleeps: list[float]) -> None:
    operation = Flaky()

    assert call_with_retry(operation, policy(), sleep=sleeps.append) == "ok"
    assert operation.calls == 1 and sleeps == []


def test_recovers_after_retryable_failures_with_backoff(sleeps: list[float]) -> None:
    operation = Flaky(ConnectionError("1"), ConnectionError("2"))
    retries: list[tuple[int, str, float]] = []

    result = call_with_retry(
        operation, policy(), sleep=sleeps.append, on_retry=lambda n, e, d: retries.append((n, str(e), d))
    )

    assert result == "ok"
    assert operation.calls == 3
    assert sleeps == [0.5, 1.0]
    assert retries == [(1, "1", 0.5), (2, "2", 1.0)]


def test_non_retryable_errors_propagate_immediately(sleeps: list[float]) -> None:
    operation = Flaky(ValueError("bad input"))

    with pytest.raises(ValueError, match="bad input"):
        call_with_retry(operation, policy(), sleep=sleeps.append)
    assert operation.calls == 1 and sleeps == []


def test_assertion_failures_are_not_retried(sleeps: list[float]) -> None:
    operation = Flaky(AssertionError("check failed"))

    with pytest.raises(AssertionError):
        call_with_retry(operation, policy(), sleep=sleeps.append)
    assert operation.calls == 1


def test_attempts_are_bounded_and_the_last_error_is_kept(sleeps: list[float]) -> None:
    errors = [ConnectionError(str(i)) for i in range(5)]
    operation = Flaky(*errors)

    with pytest.raises(RetryExhaustedError) as raised:
        call_with_retry(operation, policy(max_attempts=3), sleep=sleeps.append)

    assert operation.calls == 3 and len(sleeps) == 2
    assert raised.value.attempts == 3
    assert raised.value.__cause__ is errors[2]
    assert raised.value.last_error is errors[2]


def test_delay_is_capped() -> None:
    capped = policy(delay_seconds=1.0, backoff_multiplier=10.0, max_delay_seconds=5.0, max_attempts=5)

    assert [capped.delay_before(n) for n in (2, 3, 4)] == [1.0, 5.0, 5.0]


def test_jitter_is_bounded_and_reproducible() -> None:
    jittered = policy(delay_seconds=1.0, jitter_ratio=0.2)
    first = [jittered.delay_before(2, random.Random(3)) for _ in range(1)]
    delays = [jittered.delay_before(2, random.Random(seed)) for seed in range(200)]

    assert first == [jittered.delay_before(2, random.Random(3))]
    assert all(0.8 <= d <= 1.2 for d in delays)
    assert len(set(delays)) > 1


@pytest.mark.parametrize(
    "retry_on",
    [(), (Exception,), (BaseException,), (AssertionError,), ("ConnectionError",)],
)
def test_broad_or_invalid_retry_targets_are_refused(retry_on: tuple) -> None:
    with pytest.raises((ValueError, TypeError)):
        RetryPolicy(retry_on=retry_on)


@pytest.mark.parametrize(
    "overrides",
    [{"max_attempts": 0}, {"delay_seconds": -1}, {"backoff_multiplier": 0.5}, {"jitter_ratio": 2}],
)
def test_invalid_policy_values_are_refused(overrides: dict) -> None:
    with pytest.raises(ValueError):
        policy(**overrides)


def test_decorator_and_settings(sleeps: list[float]) -> None:
    operation = Flaky(TimeoutError())
    retry_policy = RetryPolicy.from_settings(
        RetrySettings(max_attempts=2, delay_seconds=0.1), retry_on=(TimeoutError,)
    )

    @retrying(retry_policy, sleep=sleeps.append)
    def fetch(value: str) -> str:
        return f"{operation()}-{value}"

    assert fetch("x") == "ok-x"
    assert sleeps == [0.1]
