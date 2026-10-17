"""Real HTTP against the local test service (127.0.0.1): no external network, deterministic."""

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from complete_automation.api import (
    ApiTimeoutError,
    AsyncApiClient,
    AuthenticationError,
    BearerTokenAuth,
    OAuth2ClientCredentials,
    ResponseDecodeError,
    RetryConfig,
)
from complete_automation.api.assertions import (
    assert_collection,
    assert_content_type,
    assert_error_response,
    assert_header,
    assert_latency,
    assert_pagination,
    assert_status,
)
from complete_automation.api.contract import OpenApiContract
from complete_automation.core.datasets import DataLoader
from tests.support.api_server import CLIENT_ID, CLIENT_SECRET, VALID_TOKEN

pytestmark = [pytest.mark.api, pytest.mark.integration, pytest.mark.cauto_module("api")]

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = OpenApiContract.from_file(ROOT / "data" / "api" / "contracts" / "users_api.openapi.yaml")


class CreateUserCase(BaseModel):
    case_id: str
    payload: dict
    expected_status: int
    expected_error_fields: list[str] = []


CREATE_CASES = DataLoader(ROOT / "data").load_records("api/users/create_user_cases.yaml", CreateUserCase)


@pytest.fixture
def client(api_client_factory, local_api):
    return api_client_factory(base_url=local_api.base_url, auth=BearerTokenAuth(VALID_TOKEN))


def test_health_get_head_options(client) -> None:
    health = client.get("/health")
    assert_status(health, 200)
    assert_content_type(health, "application/json")
    assert health.json() == {"status": "ok"}
    assert health.server_timing() == {"app": 1.5}
    assert_latency(health, 5000)
    assert client.head("/health").content == b""
    assert_header(client.options("/health"), "Allow", "GET, HEAD, OPTIONS")


def test_pagination_and_contract(client) -> None:
    first = client.get("/users", params={"page": 1, "per_page": 2})
    last = client.get("/users", params={"page": 3, "per_page": 2})

    for page in (first, last):
        assert_status(page, 200)
        assert_pagination(page)
        CONTRACT.assert_response(page, include_request=True)
    assert_collection(first, "data", min_items=2, max_items=2, unique_by="id")
    assert len(last.json()["data"]) == 1


@pytest.mark.parametrize("case", CREATE_CASES, ids=lambda case: case.case_id)
def test_create_user_cases(client, case: CreateUserCase) -> None:
    response = client.post("/users", json=case.payload)

    assert_status(response, case.expected_status)
    CONTRACT.assert_response(response)
    if case.expected_error_fields:
        assert_error_response(response, 422, fields=("error", "details"))
        assert sorted(d["field"] for d in response.json()["details"]) == sorted(case.expected_error_fields)
    else:
        assert_header(response, "Location", f"/v1/users/{response.json()['id']}")


def test_get_and_delete(client) -> None:
    assert_status(client.get("/users/{user_id}", path_params={"user_id": 1}), 200)
    assert_status(client.delete("/users/{user_id}", path_params={"user_id": 2}), 204)
    gone = client.get("/users/{user_id}", path_params={"user_id": 2})
    assert_error_response(gone, 404)
    CONTRACT.assert_response(gone)


def test_authentication_failures(client, api_client_factory, local_api) -> None:
    anonymous = client.get("/users", auth=None)
    wrong = api_client_factory(base_url=local_api.base_url, auth=BearerTokenAuth("wrong")).get("/users")

    for response in (anonymous, wrong):
        assert_error_response(response, 401)
        assert_header(response, "WWW-Authenticate", "Bearer")
        CONTRACT.assert_response(response)


def test_server_error_malformed_text_and_binary(client) -> None:
    assert_error_response(client.get("/error"), 500)
    with pytest.raises(ResponseDecodeError, match="not valid JSON"):
        client.get("/malformed").json()
    text = client.get("/text")
    assert text.text == "plain text body" and text.content_type == "text/plain"
    download = client.get("/download")
    assert download.content == bytes(range(256))
    assert_content_type(download, "application/octet-stream")


def test_upload_and_form(client) -> None:
    uploaded = client.post("/upload", files={"report": ("report.csv", b"a,b\n1,2\n", "text/csv")}).json()
    form = client.post("/form", data={"name": "Ada Lovelace", "role": "admin"}).json()

    assert uploaded["files"] == [{"field": "report", "filename": "report.csv", "size": 8, "content_type": "text/csv"}]
    assert form == {"name": "Ada Lovelace", "role": "admin"}


def test_cookies_round_trip(client) -> None:
    client.get("/cookies/set")
    echoed = client.get("/echo").json()

    assert "session=abc123" in echoed["headers"].get("Cookie", "")


def test_redirects_same_and_cross_origin(api_client_factory, local_api, other_origin) -> None:
    client = api_client_factory(base_url=local_api.base_url, auth=BearerTokenAuth(VALID_TOKEN),
                                follow_redirects=True, cookies={"session": "s-1"})

    same = client.get("/redirect", params={"to": "/v1/health"})
    away = client.get("/redirect", params={"to": f"{other_origin.base_url}/landing"})

    assert_status(same, 200)
    assert len(same.redirects) == 1 and same.redirects[0].startswith("302 ")
    received = {k.lower() for k in away.json()["headers"]}
    assert away.json()["path"] == "/landing"
    assert not received & {"authorization", "cookie"}


def test_read_timeout(client) -> None:
    with pytest.raises(ApiTimeoutError) as raised:
        client.get("/slow", params={"delay": "1.0"}, timeout=0.2)
    assert "GET /slow" in str(raised.value)


def test_retryable_statuses_and_retry_after(api_client_factory, local_api) -> None:
    waits: list[float] = []
    client = api_client_factory(base_url=local_api.base_url, retry=RetryConfig(max_attempts=3, delay_seconds=0.01),
                                sleep=waits.append)

    flaky = client.get("/flaky", params={"key": "a", "fail": 2})
    limited = client.get("/flaky", params={"key": "b", "fail": 1, "status": 429, "retry_after": 0})

    assert flaky.status_code == 200 and flaky.json()["attempts"] == 3 and flaky.retries == 2
    assert limited.status_code == 200 and waits[-1] == 0.0  # Retry-After: 0 respected


def test_idempotency_protection(api_client_factory, local_api) -> None:
    client = api_client_factory(base_url=local_api.base_url, retry=RetryConfig(max_attempts=3, delay_seconds=0),
                                sleep=lambda s: None)

    unsafe = client.post("/flaky-post", params={"key": "p1", "fail": 1})
    keyed = client.post("/flaky-post", params={"key": "p2", "fail": 1}, idempotency_key="order-7")

    assert unsafe.status_code == 503 and unsafe.retries == 0
    assert keyed.status_code == 200 and keyed.retries == 1
    assert local_api.service.counters == {"POST p1": 1, "POST p2": 2}


def test_oauth2_client_credentials_end_to_end(api_client_factory, local_api) -> None:
    auth = OAuth2ClientCredentials(f"{local_api.base_url}/token", CLIENT_ID, CLIENT_SECRET, scopes=["users:read"])
    client = api_client_factory(base_url=local_api.base_url, auth=auth)

    assert_status(client.get("/users"), 200)
    assert_status(client.get("/users/{user_id}", path_params={"user_id": 1}), 200)
    assert local_api.service.tokens_issued == 1  # cached

    bad = OAuth2ClientCredentials(f"{local_api.base_url}/token", CLIENT_ID, "wrong-secret")
    with pytest.raises(AuthenticationError, match=r"HTTP 401 \(invalid_client\)") as raised:
        api_client_factory(base_url=local_api.base_url, auth=bad).get("/users")
    assert "wrong-secret" not in str(raised.value)


def test_async_client_against_the_service(local_api) -> None:
    async def scenario() -> list[int]:
        async with AsyncApiClient(local_api.base_url, auth=BearerTokenAuth(VALID_TOKEN)) as client:
            responses = await asyncio.gather(
                *(client.get("/users/{user_id}", path_params={"user_id": i}) for i in range(1, 6))
            )
        return [r.json()["id"] for r in responses]

    assert asyncio.run(scenario()) == [1, 2, 3, 4, 5]


def test_contract_detects_the_deliberately_broken_endpoint(client) -> None:
    response = client.get("/legacy/users/{user_id}", path_params={"user_id": 1})

    locations = {v.location for v in CONTRACT.validate_response(response)}
    assert "response.body.id" in locations
    assert any("nickname" in str(v) for v in CONTRACT.validate_response(response))
