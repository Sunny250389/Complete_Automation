"""Authentication strategies, including OAuth2 client credentials with caching and refresh."""

import asyncio
import base64
from urllib.parse import parse_qs

import httpx
import pytest

from complete_automation.api import (
    ApiKeyAuth,
    AuthenticationError,
    BearerTokenAuth,
    OAuth2ClientCredentials,
    basic_auth,
    build_auth,
)
from complete_automation.api.mocking import MockApi
from complete_automation.api.settings import ApiSettings
from complete_automation.core.exceptions import ErrorCategory

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


def seen_by(mock: MockApi, auth: httpx.Auth, path: str = "/x") -> httpx.Request:
    route = mock.route("GET", path).respond(200, json={})
    with mock.client(auth=auth) as client:
        client.get(path)
    return route.calls[-1]


def test_api_key_in_header(mock_api: MockApi) -> None:
    request = seen_by(mock_api, ApiKeyAuth("k-123", name="X-Service-Key"))

    assert request.headers["X-Service-Key"] == "k-123"
    assert "k-123" not in str(request.url)


def test_api_key_in_query_when_required(mock_api: MockApi) -> None:
    request = seen_by(mock_api, ApiKeyAuth("k-123", name="api_key", location="query"))

    assert request.url.params["api_key"] == "k-123"


def test_auth_objects_do_not_reveal_secrets() -> None:
    for auth in (
        ApiKeyAuth("k-123"),
        BearerTokenAuth("tok-456"),
        OAuth2ClientCredentials("https://idp.test/token", "client", "secret-789"),
    ):
        assert not any(secret in repr(auth) for secret in ("k-123", "tok-456", "secret-789"))


def test_invalid_auth_arguments() -> None:
    with pytest.raises(ValueError):
        ApiKeyAuth("")
    with pytest.raises(ValueError):
        ApiKeyAuth("k", location="cookie")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BearerTokenAuth()
    with pytest.raises(ValueError):
        BearerTokenAuth("t", token_provider=lambda: "t")


def test_bearer_token_fixed_and_provided(mock_api: MockApi) -> None:
    assert seen_by(mock_api, BearerTokenAuth("tok-1")).headers["Authorization"] == "Bearer tok-1"
    tokens = iter(["first", "second"])
    provider = BearerTokenAuth(token_provider=lambda: next(tokens))
    assert seen_by(mock_api, provider, "/a").headers["Authorization"] == "Bearer first"
    assert seen_by(mock_api, provider, "/b").headers["Authorization"] == "Bearer second"


def test_bearer_refresh_on_401_happens_once(mock_api: MockApi) -> None:
    refreshed = []

    def handler(request: httpx.Request) -> httpx.Response:
        ok = request.headers.get("Authorization") == "Bearer fresh"
        return httpx.Response(200 if ok else 401, json={})

    route = mock_api.route("GET", "/secure").handle(handler)
    auth = BearerTokenAuth("stale", refresh=lambda: refreshed.append(1) or "fresh")
    with mock_api.client(auth=auth) as client:
        assert client.get("/secure").status_code == 200
        assert client.get("/secure").status_code == 200  # the fresh token is kept

    assert len(refreshed) == 1 and len(route.calls) == 3


def test_basic_auth_uses_standard_encoding(mock_api: MockApi) -> None:
    request = seen_by(mock_api, basic_auth("user", "p@ss:word"))

    assert request.headers["Authorization"] == "Basic " + base64.b64encode(b"user:p@ss:word").decode()


def test_requests_can_opt_out_of_authentication(mock_api: MockApi) -> None:
    route = mock_api.route("GET", "/x").respond(200)
    with mock_api.client(auth=BearerTokenAuth("t")) as client:
        client.get("/x", auth=None)

    assert "Authorization" not in route.calls[-1].headers


class TokenEndpoint:
    """A fake OAuth2 token endpoint."""

    def __init__(self, status: int = 200, expires_in: int | None = 3600, body: dict | None = None) -> None:
        self.status, self.expires_in, self.body = status, expires_in, body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.body is not None:
            return httpx.Response(self.status, json=self.body)
        payload = {"access_token": f"token-{len(self.requests)}", "token_type": "bearer"}
        if self.expires_in is not None:
            payload["expires_in"] = self.expires_in
        return httpx.Response(self.status, json=payload)


def oauth(endpoint: TokenEndpoint, **kwargs) -> OAuth2ClientCredentials:
    transport = httpx.MockTransport(endpoint)
    return OAuth2ClientCredentials(
        "https://idp.example.test/token", "client-1", "secret-1", transport=transport,
        async_transport=transport, **kwargs,
    )


def test_oauth2_token_request_basic_client_auth_and_scopes(mock_api: MockApi) -> None:
    endpoint = TokenEndpoint()
    request = seen_by(mock_api, oauth(endpoint, scopes=["read", "write"], audience="api://users"))

    token_request = endpoint.requests[0]
    form = {k: v[0] for k, v in parse_qs(token_request.content.decode()).items()}
    assert form == {"grant_type": "client_credentials", "scope": "read write", "audience": "api://users"}
    assert token_request.headers["Authorization"] == "Basic " + base64.b64encode(b"client-1:secret-1").decode()
    assert request.headers["Authorization"] == "Bearer token-1"


def test_oauth2_client_secret_in_body(mock_api: MockApi) -> None:
    endpoint = TokenEndpoint()
    seen_by(mock_api, oauth(endpoint, client_auth="body"))

    form = parse_qs(endpoint.requests[0].content.decode())
    assert form["client_id"] == ["client-1"] and form["client_secret"] == ["secret-1"]
    assert "Authorization" not in endpoint.requests[0].headers


def test_oauth2_token_is_cached_until_expiry(mock_api: MockApi) -> None:
    now = [1000.0]
    endpoint = TokenEndpoint(expires_in=100)
    auth = oauth(endpoint, leeway_seconds=10, clock=lambda: now[0])
    route = mock_api.route("GET", "/x").respond(200)

    with mock_api.client(auth=auth) as client:
        client.get("/x")
        client.get("/x")
        now[0] += 89  # still inside expires_in - leeway
        client.get("/x")
        now[0] += 2  # past it
        client.get("/x")

    assert len(endpoint.requests) == 2
    assert [c.headers["Authorization"] for c in route.calls] == ["Bearer token-1"] * 3 + ["Bearer token-2"]


def test_oauth2_reacquires_once_after_401(mock_api: MockApi) -> None:
    endpoint = TokenEndpoint()
    route = mock_api.route("GET", "/x").handle(
        lambda r: httpx.Response(401 if r.headers["Authorization"] == "Bearer token-1" else 200)
    )

    with mock_api.client(auth=oauth(endpoint)) as client:
        assert client.get("/x").status_code == 200

    assert len(endpoint.requests) == 2 and len(route.calls) == 2


@pytest.mark.parametrize(
    ("status", "body", "category", "text"),
    [
        (401, {"error": "invalid_client"}, ErrorCategory.CONFIGURATION, "HTTP 401 (invalid_client)"),
        (503, {"error": "temporarily_unavailable"}, ErrorCategory.INFRASTRUCTURE, "HTTP 503"),
        (200, {"token_type": "bearer"}, ErrorCategory.CONFIGURATION, "no access_token"),
        (200, {"access_token": "t", "token_type": "mac"}, ErrorCategory.CONFIGURATION, "token_type"),
    ],
)
def test_oauth2_failures_are_sanitized_and_classified(
    mock_api: MockApi, status: int, body: dict, category: ErrorCategory, text: str
) -> None:
    mock_api.route("GET", "/x").respond(200)
    with mock_api.client(auth=oauth(TokenEndpoint(status=status, body=body))) as client:
        with pytest.raises(AuthenticationError) as raised:
            client.get("/x")

    assert text in str(raised.value)
    assert raised.value.category is category
    assert "secret-1" not in str(raised.value)


def test_oauth2_transport_failure_is_infrastructure(mock_api: MockApi) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    auth = OAuth2ClientCredentials("https://idp.example.test/token", "c", "s", transport=httpx.MockTransport(refuse))
    mock_api.route("GET", "/x").respond(200)
    with mock_api.client(auth=auth) as client, pytest.raises(AuthenticationError) as raised:
        client.get("/x")

    assert raised.value.category is ErrorCategory.INFRASTRUCTURE
    assert isinstance(raised.value.__cause__, httpx.ConnectError)


def test_oauth2_async_flow_shares_the_cache(mock_api: MockApi) -> None:
    endpoint = TokenEndpoint()
    auth = oauth(endpoint)
    route = mock_api.route("GET", "/x").respond(200)

    async def scenario() -> None:
        async with mock_api.async_client(auth=auth) as client:
            await asyncio.gather(*(client.get("/x") for _ in range(5)))

    asyncio.run(scenario())
    assert len(endpoint.requests) == 1
    assert {c.headers["Authorization"] for c in route.calls} == {"Bearer token-1"}


def test_build_auth_from_settings() -> None:
    def from_settings(auth: dict) -> httpx.Auth | None:
        return build_auth(ApiSettings.model_validate({"auth": auth}).auth)

    assert from_settings(None) is None  # type: ignore[arg-type]
    assert isinstance(from_settings({"type": "api_key", "value": "k"}), ApiKeyAuth)
    assert isinstance(from_settings({"type": "bearer", "token": "t"}), BearerTokenAuth)
    assert isinstance(from_settings({"type": "basic", "username": "u", "password": "p"}), httpx.BasicAuth)
    oauth2 = from_settings({"type": "oauth2_client_credentials", "token_url": "https://idp/t",
                            "client_id": "c", "client_secret": "s", "scopes": ["a"]})
    assert isinstance(oauth2, OAuth2ClientCredentials) and oauth2.scopes == ("a",)
