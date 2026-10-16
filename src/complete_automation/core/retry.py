"""Bounded retries for operations the caller declares retryable.

Nothing is retried by default: a :class:`RetryPolicy` must name the exception
types to retry, and broad bases (``Exception``, ``BaseException``) and
``AssertionError`` are refused. Only retry operations that are safe to repeat
(idempotent or side-effect free); that judgement belongs to the caller.

``sleep`` and ``rng`` are injectable so tests run instantly and deterministically.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from complete_automation.core.config import RetrySettings
from complete_automation.core.exceptions import RetryExhaustedError

P = ParamSpec("P")
R = TypeVar("R")

_FORBIDDEN = (BaseException, Exception, AssertionError)


@dataclass(frozen=True)
class RetryPolicy:
    retry_on: tuple[type[BaseException], ...]
    max_attempts: int = 3
    delay_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 10.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not self.retry_on:
            raise ValueError("retry_on must name at least one exception type")
        for kind in self.retry_on:
            if not (isinstance(kind, type) and issubclass(kind, BaseException)):
                raise TypeError(f"retry_on entries must be exception types, got {kind!r}")
            if kind in _FORBIDDEN:
                raise ValueError(f"refusing to retry on {kind.__name__}: name specific errors")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("delays must not be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    @classmethod
    def from_settings(
        cls, settings: RetrySettings, retry_on: tuple[type[BaseException], ...]
    ) -> RetryPolicy:
        return cls(
            retry_on=retry_on,
            max_attempts=settings.max_attempts,
            delay_seconds=settings.delay_seconds,
            backoff_multiplier=settings.backoff_multiplier,
            max_delay_seconds=settings.max_delay_seconds,
            jitter_ratio=settings.jitter_ratio,
        )

    def delay_before(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay to wait before ``attempt`` (2, 3, ...), capped and optionally jittered."""
        base = min(
            self.delay_seconds * self.backoff_multiplier ** (attempt - 2), self.max_delay_seconds
        )
        if self.jitter_ratio and base:
            spread = base * self.jitter_ratio
            base = max(0.0, base + (rng or random).uniform(-spread, spread))
        return base


def call_with_retry(
    operation: Callable[[], R],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> R:
    """Run ``operation`` until it succeeds or the policy is exhausted.

    Errors not listed in ``policy.retry_on`` propagate immediately, unwrapped.
    When every attempt fails, :class:`RetryExhaustedError` is raised with the
    last error chained.
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except policy.retry_on as error:
            if attempt == policy.max_attempts:
                raise RetryExhaustedError(attempt, error) from error
            delay = policy.delay_before(attempt + 1, rng)
            if on_retry is not None:
                on_retry(attempt, error, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def retrying(
    policy: RetryPolicy, **options: object
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of :func:`call_with_retry`."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return call_with_retry(lambda: function(*args, **kwargs), policy, **options)  # type: ignore[arg-type]

        return wrapper

    return decorate
