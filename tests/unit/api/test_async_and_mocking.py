"""AsyncApiClient behaviour and the MockApi service virtualization helpers."""

import asyncio

import httpx
import pytest

from complete_automation.api import ApiTimeoutError, RetryConfig, UnsafeOperationError
from complete_automation.api.mocking import MockApi, Reply

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


def test_async_requests_run_concurrently(mock_api: MockApi) -> None:
    route = mock_api.route("GET", "/users/{user_id}").handle(
        lambda r: httpx.Response(200, json={"id": int(r.url.path.rsplit("/", 1)[1])})
    )

    async def scenario() -> list[int]:
        async with mock_api.async_client() as client:
            responses = await asyncio.gather(
                *(client.get("/users/{user_id}", path_params={"user_id": i}) for i in range(1, 6))
            )
            return [r.json()["id"] for r in responses]

    assert asyncio.run(scenario()) == [1, 2, 3, 4, 5]
    assert len(route.calls) == 5


def test_async_post_retry_timeout_and_read_only(mock_api: MockApi) -> None:
    mock_api.route("POST", "/items").respond(201, json={"id": 1})
    mock_api.route("GET", "/flaky").respond_sequence([Reply(503), Reply(200)])
    mock_api.route("GET", "/slow").fail_with(httpx.ReadTimeout)
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    async def scenario() -> None:
        async with mock_api.async_client(retry=RetryConfig(delay_seconds=0.2), sleep=fake_sleep) as client:
            assert (await client.post("/items", json={"name": "a"})).status_code == 201
            flaky = await client.get("/flaky")
            assert flaky.status_code == 200 and flaky.retries == 1
            with pytest.raises(ApiTimeoutError):
                await client.get("/slow", retry=None)
        async with mock_api.async_client(read_only=True) as read_only:
            with pytest.raises(UnsafeOperationError):
                await read_only.delete("/items")

    asyncio.run(scenario())
    assert waits == [0.2]


def test_async_client_lifecycle(mock_api: MockApi) -> None:
    async def scenario() -> bool:
        client = mock_api.async_client()
        async with client:
            assert not client.is_closed
        return client.is_closed

    assert asyncio.run(scenario()) is True


def test_mock_routes_templates_sequences_and_unmatched(mock_api: MockApi) -> None:
    users = mock_api.route("GET", "/users/{user_id}").respond(200, json={"ok": True})
    mock_api.route("GET", "/seq").respond_sequence([Reply(500), Reply(200, text="done")])

    with mock_api.client() as client:
        assert client.get("/users/{user_id}", path_params={"user_id": 5}).status_code == 200
        assert [client.get("/seq").status_code for _ in range(3)] == [500, 200, 200]  # the last reply repeats
        missing = client.get("/nowhere")

    assert len(users.calls) == 1 and users.calls[0].url.path == "/users/5"
    assert missing.status_code == 501
    with pytest.raises(AssertionError, match="GET /nowhere"):
        mock_api.assert_no_unmatched()


def test_later_routes_override_earlier_ones(mock_api: MockApi) -> None:
    mock_api.route("GET", "/x").respond(200)
    mock_api.route("GET", "/x").respond(204)

    with mock_api.client() as client:
        assert client.get("/x").status_code == 204


def test_empty_sequence_is_rejected(mock_api: MockApi) -> None:
    with pytest.raises(ValueError):
        mock_api.route("GET", "/x").respond_sequence([])
