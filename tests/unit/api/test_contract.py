"""OpenAPI contract validation against the example contract, with mocked responses."""

import copy
from pathlib import Path

import httpx
import pytest
import yaml

from complete_automation.api import ContractError
from complete_automation.api.contract import OpenApiContract, find_breaking_changes
from complete_automation.api.mocking import MockApi

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]

SPEC_PATH = Path(__file__).resolve().parents[3] / "data" / "api" / "contracts" / "users_api.openapi.yaml"
USER = {"id": 1, "name": "Ada", "email": "ada@example.test", "role": "admin", "created_at": "2026-01-01T09:00:00Z"}


@pytest.fixture(scope="module")
def contract() -> OpenApiContract:
    return OpenApiContract.from_file(SPEC_PATH)


@pytest.fixture
def api() -> MockApi:
    return MockApi("https://api.example.test/v1")


def test_operations_and_lookup(contract: OpenApiContract) -> None:
    assert ("GET", "/users/{user_id}") in contract.operations()
    found = contract.find_operation("GET", "/v1/users/42?x=1")
    assert found is not None and found.path == "/users/{user_id}" and found.concrete == "/users/42"
    assert contract.find_operation("GET", "/users") is not None  # without the server base path
    assert contract.find_operation("PATCH", "/v1/users/1") is None


def test_conforming_response_passes(contract: OpenApiContract, api: MockApi) -> None:
    api.route("GET", "/users/{user_id}").respond(200, json=USER)
    with api.client() as client:
        response = client.get("/users/{user_id}", path_params={"user_id": 1})

    assert contract.validate_response(response) == []
    assert contract.validate_exchange(response) == []
    contract.assert_response(response, include_request=True)


def test_schema_mismatch_is_reported(contract: OpenApiContract, api: MockApi) -> None:
    api.route("GET", "/users/{user_id}").respond(200, json={**USER, "id": "1", "role": "owner", "extra": True})
    with api.client() as client:
        response = client.get("/users/{user_id}", path_params={"user_id": 1})

    messages = [str(v) for v in contract.validate_response(response)]
    assert any(m.startswith("response.body.id:") and "integer" in m for m in messages)
    assert any(m.startswith("response.body.role:") for m in messages)
    assert any("extra" in m for m in messages)  # additionalProperties: false
    with pytest.raises(AssertionError, match="breaks the contract"):
        contract.assert_response(response)


def test_undocumented_status_and_content_type(contract: OpenApiContract, api: MockApi) -> None:
    api.route("GET", "/users/{user_id}").respond(418, json={"error": "teapot"})
    api.route("GET", "/health").respond(200, text="ok", headers={"Content-Type": "text/plain"})
    with api.client() as client:
        teapot = client.get("/users/{user_id}", path_params={"user_id": 1})
        plain = client.get("/health")

    assert "418 is not documented" in str(contract.validate_response(teapot)[0])
    assert "content type text/plain is not documented" in str(contract.validate_response(plain)[0])


def test_documented_error_status_and_required_header(contract: OpenApiContract, api: MockApi) -> None:
    api.route("POST", "/users").respond(201, json=USER)  # Location header missing
    api.route("GET", "/users/{user_id}").respond(404, json={"error": "user not found"})
    with api.client() as client:
        created = client.post("/users", json={"name": "Ada", "email": "ada@example.test"})
        missing = client.get("/users/{user_id}", path_params={"user_id": 99})

    assert [str(v) for v in contract.validate_response(created)] == ["response.header.Location: required header is missing"]
    assert contract.validate_response(missing) == []


def test_request_validation(contract: OpenApiContract) -> None:
    ok = contract.validate_request("POST", "/v1/users", body={"name": "Ada", "email": "a@b.test"},
                                   content_type="application/json")
    assert ok == []

    missing_body = contract.validate_request("POST", "/v1/users", content_type="application/json")
    assert [str(v) for v in missing_body] == ["request.body: required body is missing"]

    bad_body = contract.validate_request("POST", "/v1/users", body={"email": "nope", "role": "owner"},
                                         content_type="application/json")
    assert {v.location for v in bad_body} >= {"request.body", "request.body.email", "request.body.role"}

    wrong_type = contract.validate_request("POST", "/v1/users", body="name=Ada",
                                           content_type="application/x-www-form-urlencoded")
    assert "content type application/x-www-form-urlencoded is not documented" in str(wrong_type[0])

    bad_params = contract.validate_request("GET", "/v1/users", query={"page": "0", "per_page": "many"})
    assert {v.location for v in bad_params} == {"request.query.page", "request.query.per_page"}

    bad_path = contract.validate_request("GET", "/v1/users/abc")
    assert [v.location for v in bad_path] == ["request.path.user_id"]

    assert "not documented" in str(contract.validate_request("GET", "/v1/nowhere")[0])


def test_openapi_30_nullable_is_understood() -> None:
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "t", "version": "1"},
        "paths": {"/thing": {"get": {"responses": {"200": {"description": "ok", "content": {"application/json": {
            "schema": {"type": "object", "properties": {"note": {"type": "string", "nullable": True}}}}}}}}}},
    }
    contract = OpenApiContract(spec)

    ok = contract.validate_response(method="GET", path="/thing", status_code=200, body={"note": None},
                                    content_type="application/json")
    bad = contract.validate_response(method="GET", path="/thing", status_code=200, body={"note": 5},
                                     content_type="application/json")
    assert ok == [] and len(bad) == 1


def test_invalid_and_unsupported_documents(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="invalid OpenAPI document"):
        OpenApiContract({"openapi": "3.1.0", "info": {"title": "no version"}, "paths": {}})
    with pytest.raises(ContractError):
        OpenApiContract({"swagger": "2.0", "info": {"title": "t", "version": "1"}, "paths": {}})
    with pytest.raises(ContractError, match="cannot load"):
        OpenApiContract.from_file(tmp_path / "missing.yaml")


def test_remote_documents_are_fetched_explicitly() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.yaml":
            return httpx.Response(200, text=text, headers={"Content-Type": "application/yaml"})
        return httpx.Response(404)

    transport = httpx.MockTransport(serve)
    contract = OpenApiContract.from_url("https://specs.example.test/openapi.yaml", transport=transport, timeout=2)
    assert contract.source == "https://specs.example.test/openapi.yaml"
    with pytest.raises(ContractError, match="HTTP 404"):
        OpenApiContract.from_url("https://specs.example.test/missing.yaml", transport=transport)

    def hang(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(ContractError, match="ConnectTimeout"):
        OpenApiContract.from_url("https://specs.example.test/openapi.yaml", transport=httpx.MockTransport(hang))


def test_breaking_changes_are_detected(contract: OpenApiContract) -> None:
    spec = copy.deepcopy(yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8")))
    del spec["paths"]["/users/{user_id}"]["delete"]
    del spec["paths"]["/users"]["get"]["responses"]["422"]
    spec["paths"]["/users"]["get"]["parameters"].append({"name": "tenant", "in": "query", "required": True,
                                                          "schema": {"type": "string"}})
    spec["components"]["schemas"]["NewUser"]["required"].append("role")
    del spec["components"]["schemas"]["User"]["properties"]["created_at"]
    spec["components"]["schemas"]["User"]["required"].remove("created_at")
    changed = OpenApiContract(spec)

    changes = find_breaking_changes(contract, changed)

    assert "DELETE /users/{user_id}: operation removed" in changes
    assert "GET /users: response 422 no longer documented" in changes
    assert "GET /users: new required query parameter 'tenant'" in changes
    assert "POST /users: request property 'role' became required" in changes
    assert "GET /users/{user_id}: response 200 no longer returns 'created_at'" in changes
    assert find_breaking_changes(contract, contract) == []
