"""Redirect safety and sanitized diagnostics."""

import json

import httpx
import pytest

from complete_automation.api import ApiKeyAuth, BearerTokenAuth, RedirectLoopError
from complete_automation.api.diagnostics import Sanitizer
from complete_automation.api.mocking import MockApi

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


def two_origins() -> tuple[MockApi, list[httpx.Request]]:
    """One transport serving api.example.test (redirecting) and other.example (recording)."""
    other_calls: list[httpx.Request] = []
    mock = MockApi("https://api.example.test")
    mock.route("GET", "/same").respond(302, headers={"Location": "/target"})
    mock.route("GET", "/away").respond(302, headers={"Location": "https://other.example/landing"})
    mock.route("GET", "/downgrade").respond(301, headers={"Location": "http://api.example.test/target"})
    mock.route("GET", "/loop").respond(302, headers={"Location": "/loop"})
    mock.route("GET", "/target").respond(200, json={"here": True})

    original = mock._dispatch

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.host == "other.example" or request.url.scheme == "http":
            other_calls.append(request)
            return httpx.Response(200, json={}, request=request)
        return original(request)

    mock._dispatch = dispatch  # type: ignore[method-assign]
    return mock, other_calls


def test_redirects_are_not_followed_by_default() -> None:
    mock, _ = two_origins()
    with mock.client() as client:
        response = client.get("/same")

    assert response.status_code == 302 and response.redirects == ()


def test_same_origin_redirect_keeps_credentials() -> None:
    mock, _ = two_origins()
    with mock.client(auth=BearerTokenAuth("tok"), follow_redirects=True) as client:
        response = client.get("/same")

    assert response.status_code == 200
    assert response.redirects == ("302 https://api.example.test/same",)
    assert response.raw.request.headers["Authorization"] == "Bearer tok"


@pytest.mark.parametrize("path", ["/away", "/downgrade"])
def test_credentials_never_follow_a_redirect_to_another_origin(path: str) -> None:
    mock, other_calls = two_origins()
    auth = ApiKeyAuth("k-secret", name="X-Service-Key")
    with mock.client(auth=auth, follow_redirects=True, headers={"X-Api-Token": "t-secret"},
                     cookies={"session": "c-secret"}) as client:
        client.get(path, headers={"Authorization": "Bearer explicit"})

    sent = other_calls[-1].headers
    for name in ("Authorization", "X-Service-Key", "X-Api-Token", "Cookie"):
        assert name not in sent, name


def test_redirect_loops_stop_at_the_limit() -> None:
    mock, _ = two_origins()
    with mock.client(follow_redirects=True, max_redirects=3) as client, pytest.raises(RedirectLoopError, match="more than 3"):
        client.get("/loop")


def test_sanitizer_urls_and_headers() -> None:
    sanitizer = Sanitizer(sensitive_names=["X-Service-Key"])

    assert sanitizer.url("https://bob:pw@h.test/p?api_key=a&page=2&access_token=b") == (
        "https://bob:***@h.test/p?api_key=***&page=2&access_token=***"
    )
    assert sanitizer.url("https://ghp_TOKEN@h.test/") == "https://***@h.test/"
    assert sanitizer.headers(
        httpx.Headers({"Authorization": "Bearer t", "X-Service-Key": "k", "Cookie": "s=1", "Accept": "a/b"})
    ) == {"authorization": "***", "x-service-key": "***", "cookie": "***", "accept": "a/b"}


def test_body_excerpts_are_redacted_and_bounded() -> None:
    sanitizer = Sanitizer()
    body = json.dumps({"user": "bob", "password": "hunter2", "items": list(range(500))}).encode()

    excerpt = sanitizer.body(body, "application/json", limit=100)

    assert "hunter2" not in excerpt and '"password": "***"' in excerpt
    assert "more characters" in excerpt
    assert sanitizer.body(b"token=abc123", "text/plain") == "token=***"
    assert sanitizer.body(b"\x00\x01", "application/octet-stream") == "<2 bytes of application/octet-stream>"
    assert sanitizer.body(b"", "application/json") == "<empty>"


def test_exchange_records_contain_no_secrets(mock_api: MockApi) -> None:
    mock_api.route("POST", "/login").respond(
        200, json={"access_token": "issued-token", "user": "bob"}, headers={"Set-Cookie": "session=s-1"}
    )
    with mock_api.client(auth=ApiKeyAuth("k-123")) as client:
        response = client.post("/login", params={"token": "q-1"}, json={"user": "bob", "password": "pw-1"})

    record = response.to_record()
    text = json.dumps(record)
    for secret in ("k-123", "q-1", "pw-1", "issued-token", "s-1"):
        assert secret not in text, secret
    assert json.loads(record["request"]["body"]) == {"user": "bob", "password": "***"}
    assert json.loads(record["response"]["body"]) == {"access_token": "***", "user": "bob"}


def test_messages_do_not_leak_secrets(mock_api: MockApi) -> None:
    mock_api.route("GET", "/x").respond(500, json={"error": "db password=hunter2 rejected"})
    with mock_api.client() as client:
        response = client.get("/x", params={"api_key": "k-9"})

    assert "hunter2" not in response.body_excerpt()
    assert "k-9" not in response.url and "k-9" not in repr(response)
