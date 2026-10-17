"""Representative API tests for the example Users API (data/api/contracts/users_api.openapi.yaml).

Read these as templates. Each one runs against the local test service; point
``base_url`` at a real deployment through configuration to reuse them.
"""

from pathlib import Path

import pytest

from complete_automation.api import BearerTokenAuth
from complete_automation.api.assertions import (
    assert_error_response,
    assert_field,
    assert_json_keys,
    assert_latency,
    assert_status,
)
from complete_automation.api.contract import OpenApiContract
from complete_automation.api.schema import assert_matches_schema
from tests.support.api_server import VALID_TOKEN

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]

CONTRACT = OpenApiContract.from_file(Path(__file__).resolve().parents[3] / "data" / "api" / "contracts" / "users_api.openapi.yaml")
USER_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "email", "role", "created_at"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "email": {"type": "string", "format": "email"},
        "created_at": {"type": "string", "format": "date-time"},
    },
}


@pytest.fixture
def users_api(api_client_factory, local_api):
    """An authenticated client; the factory closes it and records calls in the test result."""
    return api_client_factory(base_url=local_api.base_url, auth=BearerTokenAuth(VALID_TOKEN))


def test_successful_request(users_api) -> None:
    response = users_api.get("/users/{user_id}", path_params={"user_id": 1})

    assert_status(response, 200)
    assert_json_keys(response, ["id", "name", "email", "role", "created_at"])
    assert_field(response, "id", equals=1)
    assert_field(response, "created_at", datetime_format="iso8601")
    assert_latency(response, 2000)


def test_request_validation_failure(users_api) -> None:
    response = users_api.post("/users", json={"name": "", "email": "not-an-email"})

    assert_error_response(response, 422, fields=("error", "details"))
    assert {d["field"] for d in response.json()["details"]} == {"name", "email"}


def test_authentication_failure(users_api) -> None:
    response = users_api.get("/users", auth=None)  # send this request without credentials

    assert_error_response(response, 401)


def test_schema_mismatch_is_caught(users_api) -> None:
    """The legacy endpoint returns ``id`` as a string; schema validation must flag it."""
    response = users_api.get("/legacy/users/{user_id}", path_params={"user_id": 1})

    assert_status(response, 200)
    with pytest.raises(AssertionError, match=r"body\.id: '1' is not of type 'integer'"):
        assert_matches_schema(response, USER_SCHEMA)


def test_expected_server_error(users_api) -> None:
    response = users_api.get("/error")

    assert_error_response(response, 500)
    CONTRACT.assert_response(response)  # the 500 is documented, so it conforms


def test_contract_validation_failure_is_caught(users_api) -> None:
    """The contract documents a conforming User; the legacy endpoint breaks it."""
    response = users_api.get("/legacy/users/{user_id}", path_params={"user_id": 1})

    with pytest.raises(AssertionError, match="breaks the contract") as raised:
        CONTRACT.assert_response(response)
    assert "response.body.id" in str(raised.value)
