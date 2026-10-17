"""Request construction, response handling, errors and lifecycle, all through MockTransport."""

import json

import httpx
import pytest

from complete_automation.api import ApiClient, ApiTimeoutError, ApiTransportError, ResponseDecodeError, UnsafeOperationError
from complete_automation.api.mocking import MockApi
from complete_automation.core.context import TestContext, use_test_context
from complete_automation.core.exceptions import ConfigurationError, ErrorCategory, classify_exception

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


@pytest.fixture
def echo() -> MockApi:
    """Every route answers with a description of the request it received."""
    mock = MockApi("https://api.example.test/v1")

    def describe(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "method": request.method,
            "raw_path": request.url.raw_path.decode(),
            "params": [list(item) for item in request.url.params.multi_items()],
            "headers": {k.lower(): v for k, v in request.headers.items()},
            "body": request.content.decode("utf-8", errors="replace"),
        })

    for method in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        mock.route(method, "/{rest}").handle(describe)
        mock.route(method, "/users/{user_id}").handle(describe)
        mock.route(method, "/users/{user_id}/posts/{post_id}").handle(describe)
    return mock


def test_base_path_and_path_parameters_are_encoded(echo: MockApi) -> None:
    with echo.client() as client:
        seen = client.get("/users/{user_id}/posts/{post_id}", path_params={"user_id": "a/b c", "post_id": 7}).json()

    assert seen["raw_path"] == "/v1/users/a%2Fb%20c/posts/7"


@pytest.mark.parametrize(
    ("path_params", "message"),
    [({}, "missing parameter"), ({"user_id": 1, "extra": 2}, "unexpected parameter"), ({"user_id": None}, "is empty")],
)
def test_path_parameter_mistakes_are_rejected_before_sending(echo: MockApi, path_params: dict, message: str) -> None:
    with echo.client() as client, pytest.raises(ValueError, match=message):
        client.get("/users/{user_id}", path_params=path_params)
    assert not any(route.calls for route in echo.routes)


def test_query_headers_and_cookies(echo: MockApi) -> None:
    with echo.client(headers={"X-Client": "tests"}) as client:
        seen = client.get(
            "/search",
            params=[("tag", "a"), ("tag", "b"), ("q", "x y")],
            headers={"Accept": "application/json"},
            cookies={"session": "s1", "theme": "dark"},
        ).json()

    assert seen["params"] == [["tag", "a"], ["tag", "b"], ["q", "x y"]]
    assert seen["headers"]["x-client"] == "tests"
    assert seen["headers"]["accept"] == "application/json"
    assert seen["headers"]["cookie"] == "session=s1; theme=dark"


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_every_method(echo: MockApi, method: str) -> None:
    with echo.client() as client:
        response = getattr(client, method.lower())("/things")

    assert response.json()["method"] == method
    assert response.method == method


def test_head_returns_headers_without_decoding(echo: MockApi) -> None:
    with echo.client() as client:
        assert client.head("/things").status_code == 200


def test_json_form_multipart_and_raw_bodies(echo: MockApi) -> None:
    with echo.client() as client:
        as_json = client.post("/things", json={"name": "a", "n": 1}).json()
        as_form = client.post("/things", data={"name": "a b"}).json()
        as_multipart = client.post("/things", files={"file": ("report.csv", b"a,b\n1,2\n", "text/csv")}).json()
        as_raw = client.put("/things", content=b"\x00\x01", headers={"Content-Type": "application/octet-stream"}).json()

    assert json.loads(as_json["body"]) == {"name": "a", "n": 1}
    assert as_json["headers"]["content-type"] == "application/json"
    assert as_form["body"] == "name=a+b"
    assert as_form["headers"]["content-type"] == "application/x-www-form-urlencoded"
    assert as_multipart["headers"]["content-type"].startswith("multipart/form-data; boundary=")
    assert 'filename="report.csv"' in as_multipart["body"]
    assert as_raw["headers"]["content-type"] == "application/octet-stream"


def test_correlation_header_follows_the_test_context(echo: MockApi) -> None:
    context = TestContext(run_id="r", test_id="t", name="t", module="api", correlation_id="corr-123")
    with echo.client() as client:
        with use_test_context(context):
            inside = client.get("/x").json()["headers"]["x-correlation-id"]
        outside = client.get("/x").json()["headers"]["x-correlation-id"]
        explicit = client.get("/x", headers={"X-Correlation-ID": "mine"}).json()["headers"]["x-correlation-id"]

    assert inside == "corr-123"
    assert outside and outside != "corr-123"
    assert explicit == "mine"


def test_idempotency_key_header(echo: MockApi) -> None:
    with echo.client() as client:
        assert client.post("/x", idempotency_key="k-1").json()["headers"]["idempotency-key"] == "k-1"


def test_http_errors_are_returned_not_raised(mock_api: MockApi) -> None:
    mock_api.route("GET", "/missing").respond(404, json={"error": "nope"})
    mock_api.route("GET", "/broken").respond(500, json={"error": "boom"})

    with mock_api.client() as client:
        assert client.get("/missing").status_code == 404
        assert client.get("/broken").status_code == 500


def test_response_helpers(mock_api: MockApi) -> None:
    mock_api.route("GET", "/json").respond(
        200, json={"a": 1}, headers={"Content-Type": "application/problem+json", "Server-Timing": "db;dur=12.5, app;dur=3"}
    )
    mock_api.route("GET", "/text").respond(200, text="hello", headers={"Content-Type": "text/plain; charset=utf-8"})
    mock_api.route("GET", "/binary").respond(200, content=bytes(range(256)), headers={"Content-Type": "image/png"})

    with mock_api.client() as client:
        as_json, as_text, as_binary = client.get("/json"), client.get("/text"), client.get("/binary")

    assert as_json.is_json and as_json.json() == {"a": 1}
    assert as_json.content_type == "application/problem+json"
    assert as_json.server_timing() == {"db": 12.5, "app": 3.0}
    assert not as_text.is_json and as_text.text == "hello" and as_text.content_type == "text/plain"
    assert as_binary.content == bytes(range(256))
    assert as_binary.body_excerpt() == "<256 bytes of image/png>"
    assert as_json.raw.request.url.path == "/json"  # the httpx response stays reachable


def test_timing_is_measured(mock_api: MockApi) -> None:
    mock_api.route("GET", "/x").respond(200, json={})

    with mock_api.client() as client:
        response = client.get("/x")

    assert response.latency_ms >= 0 and response.total_ms >= response.latency_ms
    assert response.retries == 0 and len(response.attempts) == 1


@pytest.mark.parametrize(
    ("body", "content_type", "message"),
    [(b"", "application/json", "empty body"), (b'{"a": ', "application/json", "not valid JSON")],
)
def test_json_decoding_failures_are_assertion_errors(mock_api: MockApi, body: bytes, content_type: str, message: str) -> None:
    mock_api.route("GET", "/x").respond(200, content=body, headers={"Content-Type": content_type})

    with mock_api.client() as client:
        response = client.get("/x")
    with pytest.raises(ResponseDecodeError, match=message) as raised:
        response.json()

    assert isinstance(raised.value, AssertionError)
    assert "GET /x -> 200" in str(raised.value)
    assert classify_exception(raised.value) is ErrorCategory.ASSERTION


def test_transport_errors_are_wrapped_and_classified(mock_api: MockApi) -> None:
    mock_api.route("GET", "/down").fail_with(httpx.ConnectError)
    mock_api.route("GET", "/slow").fail_with(httpx.ReadTimeout)

    with mock_api.client() as client:
        with pytest.raises(ApiTransportError) as down:
            client.get("/down", params={"api_key": "leak-me"})
        with pytest.raises(ApiTimeoutError) as slow:
            client.get("/slow")

    assert isinstance(down.value.__cause__, httpx.ConnectError)
    assert classify_exception(down.value) is ErrorCategory.INFRASTRUCTURE
    assert classify_exception(slow.value) is ErrorCategory.TIMEOUT
    assert "GET /down" in str(down.value) and down.value.attempts == 1
    assert "leak-me" not in str(down.value)


def test_per_request_timeout_is_applied(echo: MockApi) -> None:
    with echo.client(timeout=httpx.Timeout(7.0)) as client:
        default = client.get("/x").raw.request.extensions["timeout"]
        override = client.get("/x", timeout=1.5).raw.request.extensions["timeout"]

    assert default["read"] == 7.0
    assert override["read"] == 1.5


def test_read_only_clients_refuse_mutations(echo: MockApi) -> None:
    with echo.client(read_only=True) as client:
        assert client.get("/x").status_code == 200
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with pytest.raises(UnsafeOperationError, match="read-only"):
                client.request(method, "/x")


def test_absolute_urls_must_stay_on_the_base_origin(echo: MockApi) -> None:
    with echo.client() as client:
        assert client.get("https://api.example.test/v1/x").status_code == 200
        with pytest.raises(UnsafeOperationError, match="outside"):
            client.get("https://evil.example/collect")


def test_invalid_client_configuration() -> None:
    with pytest.raises(ConfigurationError, match="absolute http"):
        ApiClient("api.example.test")
    with pytest.raises(ConfigurationError, match="finite"):
        ApiClient("https://api.example.test", timeout=None)  # type: ignore[arg-type]


def test_client_lifecycle(mock_api: MockApi) -> None:
    mock_api.route("GET", "/x").respond(200)
    client = mock_api.client()
    with client:
        assert not client.is_closed
        client.get("/x")
    assert client.is_closed
    with pytest.raises(RuntimeError):
        client.get("/x")


def test_no_network_access_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("network access during construction")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    ApiClient("https://api.example.test").close()
