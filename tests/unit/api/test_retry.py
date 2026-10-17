"""Retry and Retry-After behaviour, with an injected sleep (no real waiting)."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from complete_automation.api import ApiTransportError, RetryConfig
from complete_automation.api.client import parse_retry_after
from complete_automation.api.mocking import MockApi, Reply
from complete_automation.api.settings import ApiRetrySettings

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


@pytest.fixture
def sleeps() -> list[float]:
    return []


def test_retries_are_off_by_default(mock_api: MockApi) -> None:
    route = mock_api.route("GET", "/flaky").respond_sequence([Reply(503), Reply(200)])

    with mock_api.client() as client:
        assert client.get("/flaky").status_code == 503
    assert len(route.calls) == 1


def test_retryable_status_is_retried_with_backoff(mock_api: MockApi, sleeps: list[float]) -> None:
    route = mock_api.route("GET", "/flaky").respond_sequence([Reply(503), Reply(502), Reply(200, json={"ok": True})])

    with mock_api.client(retry=RetryConfig(delay_seconds=0.5), sleep=sleeps.append) as client:
        response = client.get("/flaky")

    assert response.status_code == 200 and response.retries == 2
    assert [a.status_code for a in response.attempts] == [503, 502, 200]
    assert sleeps == [0.5, 1.0]
    assert [a.wait_ms for a in response.attempts] == [500.0, 1000.0, 0.0]
    assert len(route.calls) == 3


def test_retry_after_seconds_and_dates_are_respected_and_capped(mock_api: MockApi, sleeps: list[float]) -> None:
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True)
    mock_api.route("GET", "/limited").respond_sequence(
        [Reply(429, headers={"Retry-After": "3"}), Reply(429, headers={"Retry-After": later}), Reply(200)]
    )

    config = RetryConfig(max_attempts=3, max_retry_after_seconds=10)
    with mock_api.client(retry=config, sleep=sleeps.append) as client:
        assert client.get("/limited").status_code == 200

    assert sleeps == [3.0, 10.0]


def test_parse_retry_after() -> None:
    now = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)

    assert parse_retry_after("5") == 5.0
    assert parse_retry_after(format_datetime(now + timedelta(seconds=30), usegmt=True), now=now) == 30.0
    assert parse_retry_after(format_datetime(now - timedelta(seconds=30), usegmt=True), now=now) == 0.0
    assert parse_retry_after("soon") is None and parse_retry_after(None) is None


def test_exhaustion_returns_the_last_response(mock_api: MockApi, sleeps: list[float]) -> None:
    mock_api.route("GET", "/down").respond(503, json={"error": "down"})

    with mock_api.client(retry=RetryConfig(max_attempts=3), sleep=sleeps.append) as client:
        response = client.get("/down")

    assert response.status_code == 503 and len(response.attempts) == 3 and len(sleeps) == 2


def test_post_is_not_retried_without_an_idempotency_mechanism(mock_api: MockApi, sleeps: list[float]) -> None:
    route = mock_api.route("POST", "/orders").respond_sequence([Reply(503), Reply(201)])

    with mock_api.client(retry=RetryConfig(), sleep=sleeps.append) as client:
        assert client.post("/orders", json={}).status_code == 503
    assert len(route.calls) == 1 and sleeps == []


@pytest.mark.parametrize("marker", [{"idempotency_key": "order-42"}, {"idempotent": True}])
def test_post_is_retried_when_the_caller_makes_it_safe(mock_api: MockApi, sleeps: list[float], marker: dict) -> None:
    route = mock_api.route("POST", "/orders").respond_sequence([Reply(503), Reply(201)])

    with mock_api.client(retry=RetryConfig(), sleep=sleeps.append) as client:
        assert client.post("/orders", json={}, **marker).status_code == 201
    assert len(route.calls) == 2
    if "idempotency_key" in marker:
        assert {c.headers["Idempotency-Key"] for c in route.calls} == {"order-42"}


def test_transport_errors_are_retried_then_raised_with_the_attempt_count(mock_api: MockApi, sleeps: list[float]) -> None:
    mock_api.route("GET", "/flaky-net").respond_sequence([Reply(error=httpx.ConnectError), Reply(200)])
    mock_api.route("GET", "/dead").fail_with(httpx.ConnectError)

    with mock_api.client(retry=RetryConfig(max_attempts=3, delay_seconds=0.1), sleep=sleeps.append) as client:
        recovered = client.get("/flaky-net")
        with pytest.raises(ApiTransportError) as raised:
            client.get("/dead")

    assert recovered.status_code == 200 and recovered.attempts[0].error == "ConnectError"
    assert raised.value.attempts == 3
    assert isinstance(raised.value.__cause__, httpx.ConnectError)


def test_per_request_override_and_disable(mock_api: MockApi, sleeps: list[float]) -> None:
    route = mock_api.route("GET", "/flaky").respond(503)

    with mock_api.client(retry=RetryConfig(max_attempts=3), sleep=sleeps.append) as client:
        client.get("/flaky", retry=None)
        assert len(route.calls) == 1
        client.get("/flaky", retry=RetryConfig(max_attempts=2))
        assert len(route.calls) == 3


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"status_codes": {401}}, "not retryable"),
        ({"status_codes": {422}}, "not retryable"),
        ({"status_codes": {200}}, "not retryable"),
        ({"methods": {"POST"}}, "idempotency key"),
        ({"methods": {"patch"}}, "idempotency key"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"delay_seconds": -1}, "negative"),
        ({"transport_errors": (ValueError,)}, "httpx.TransportError"),
    ],
)
def test_unsafe_retry_configurations_are_refused(options: dict, error: str) -> None:
    with pytest.raises((ValueError, TypeError), match=error):
        RetryConfig(**options)


def test_retry_config_from_settings() -> None:
    assert RetryConfig.from_settings(ApiRetrySettings()) is None
    config = RetryConfig.from_settings(ApiRetrySettings(enabled=True, max_attempts=4, status_codes=(503,)))
    assert config is not None and config.max_attempts == 4 and config.status_codes == frozenset({503})
