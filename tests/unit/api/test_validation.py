"""API assertions and JSON Schema validation: passing cases and failure diagnostics."""

import re
from pathlib import Path

import pytest

from complete_automation.api import ContractError
from complete_automation.api.assertions import (
    assert_collection,
    assert_content_type,
    assert_error_response,
    assert_field,
    assert_header,
    assert_headers_present,
    assert_json_keys,
    assert_json_subset,
    assert_latency,
    assert_pagination,
    assert_redirect,
    assert_status,
    assert_status_class,
    get_path,
)
from complete_automation.api.mocking import MockApi
from complete_automation.api.schema import assert_matches_schema, load_schema, schema_violations

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]

USER = {"id": 1, "name": "Ada", "email": "ada@example.test", "role": "admin", "created_at": "2026-01-01T09:00:00Z",
        "manager": None, "tags": ["a", "b"], "password_hint": "s3cr3t-hint"}
PAGE = {"page": 2, "per_page": 2, "total": 5, "total_pages": 3, "data": [{"id": 3}, {"id": 4}]}


@pytest.fixture
def responses(mock_api: MockApi):
    mock_api.route("GET", "/users/{id}").respond(200, json=USER, headers={"X-Request-ID": "req-1", "X-Api-Key": "k-1"})
    mock_api.route("GET", "/users").respond(200, json=PAGE)
    mock_api.route("POST", "/users").respond(422, json={"error": "validation failed", "details": []})
    mock_api.route("GET", "/moved").respond(301, headers={"Location": "/new-home"})
    with mock_api.client() as client:
        yield {
            "user": client.get("/users/{id}", path_params={"id": 1}),
            "page": client.get("/users"),
            "invalid": client.post("/users", json={}),
            "moved": client.get("/moved"),
        }


def failure(check) -> str:
    with pytest.raises(AssertionError) as raised:
        check()
    return str(raised.value)


def test_status_assertions(responses) -> None:
    assert_status(responses["user"], 200)
    assert_status(responses["user"], range(200, 300))
    assert_status_class(responses["invalid"], 4)

    message = failure(lambda: assert_status(responses["invalid"], 201))
    assert message.startswith("POST /users -> 422")
    assert "expected status 201, got 422" in message
    assert "validation failed" in message  # sanitized body excerpt
    assert "expected status [200, 201], got 422" in failure(lambda: assert_status(responses["invalid"], [200, 201]))


def test_latency_assertion(responses) -> None:
    assert_latency(responses["user"], 10_000)
    assert "exceeds 0.0 ms" in failure(lambda: assert_latency(responses["user"], 0.0))


def test_header_and_content_type_assertions(responses) -> None:
    user = responses["user"]
    assert_header(user, "X-Request-ID")
    assert_header(user, "X-Request-ID", "req-1")
    assert_header(user, "x-request-id", re.compile(r"^req-\d+$"))
    assert_headers_present(user, ["X-Request-ID", "Content-Type"])
    assert_content_type(user, "application/json")

    assert "missing header 'X-Trace'" in failure(lambda: assert_header(user, "X-Trace"))
    masked = failure(lambda: assert_header(user, "X-Api-Key", "other"))
    assert "k-1" not in masked and "***" in masked
    assert "expected content type 'text/csv'" in failure(lambda: assert_content_type(user, "text/csv"))


def test_redirect_assertion(responses) -> None:
    assert_redirect(responses["moved"], location="/new-home", status=301)
    assert "expected a redirect, got status 200" in failure(lambda: assert_redirect(responses["user"]))


def test_error_response_assertion(responses) -> None:
    assert_error_response(responses["invalid"], 422, fields=("error", "details"))
    assert "expected an error status" in failure(lambda: assert_error_response(responses["user"]))
    assert "lacks field(s) ['code']" in failure(lambda: assert_error_response(responses["invalid"], fields=("code",)))


def test_json_keys_and_subset(responses) -> None:
    user = responses["user"]
    assert_json_keys(user, ["id", "email"], forbidden=["password"])
    assert_json_keys({"a": 1}, ["a"], allow_extra=False)
    assert_json_subset(user, {"name": "Ada", "tags": ["a", "b"]})

    keys = failure(lambda: assert_json_keys(user, ["id", "phone"], forbidden=["password_hint"]))
    assert "missing ['phone']" in keys and "forbidden ['password_hint']" in keys
    assert "unexpected ['b']" in failure(lambda: assert_json_keys({"a": 1, "b": 2}, ["a"], allow_extra=False))
    subset = failure(lambda: assert_json_subset(user, {"name": "Grace", "role": "admin"}))
    assert "$.name: expected 'Grace', got 'Ada'" in subset


def test_field_assertions(responses) -> None:
    user = responses["user"]
    assert_field(user, "id", equals=1, type_=int, minimum=1, maximum=10)
    assert_field(user, "email", pattern=r"[^@]+@[^@]+")
    assert_field(user, "role", one_of=["admin", "member"])
    assert_field(user, "created_at", datetime_format="iso8601")
    assert_field(user, "manager", nullable=True)
    assert_field(user, "manager", equals=None)
    assert_field(user, "tags[1]", equals="b")

    assert "must not be null" in failure(lambda: assert_field(user, "manager", type_=str))
    assert "expected type str, got 1" in failure(lambda: assert_field(user, "id", type_=str))
    assert "expected >= 5" in failure(lambda: assert_field(user, "id", minimum=5))
    assert "expected a iso8601 timestamp" in failure(lambda: assert_field(user, "name", datetime_format="iso8601"))
    assert "$.phone: missing" in failure(lambda: assert_field(user, "phone", equals=1))
    hint = failure(lambda: assert_field(user, "password_hint", equals="x"))
    assert "s3cr3t-hint" not in hint
    assert "expected type int" in failure(lambda: assert_field({"flag": True}, "flag", type_=int))  # bool is not int


def test_get_path() -> None:
    assert get_path({"a": [{"b": 1}]}, "a[0].b") == 1
    assert get_path([1, 2, 3], "[-1]") == 3
    with pytest.raises(AssertionError, match=r"\$\.a\[5\]: no such item"):
        get_path({"a": []}, "a[5]")


def test_collection_and_pagination(responses) -> None:
    page = responses["page"]
    assert_collection(page, "data", min_items=1, max_items=2, contains={"id": 3}, unique_by="id",
                      each=lambda item: item["id"] > 0)
    assert_pagination(page)

    assert "at least 3 item(s), got 2" in failure(lambda: assert_collection(page, "data", min_items=3))
    assert "'id' is not unique" in failure(lambda: assert_collection([{"id": 1}, {"id": 1}], unique_by="id"))
    assert "item(s) [1] fail the check" in failure(lambda: assert_collection(page, "data", each=lambda i: i["id"] == 3))
    broken = dict(PAGE, total_pages=9)
    assert "total_pages is 9" in failure(lambda: assert_pagination(broken))
    short = dict(PAGE, data=[{"id": 3}])
    assert "page 2 should hold 2 item(s), got 1" in failure(lambda: assert_pagination(short))
    assert_pagination(dict(PAGE, page=3, data=[{"id": 5}]))  # last page holds the remainder


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["id", "email", "created_at"],
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "email": {"type": "string", "format": "email"},
        "created_at": {"type": "string", "format": "date-time"},
        "password_hint": {"type": "integer"},
    },
}


def test_schema_validation_passes_and_reports_every_violation(responses) -> None:
    assert schema_violations({"id": 1, "email": "a@b.test", "created_at": "2026-01-01T00:00:00Z"}, SCHEMA) == []

    violations = schema_violations({"id": 0, "email": "not-an-email", "created_at": "yesterday"}, SCHEMA)
    assert [v.location for v in violations] == ["body.created_at", "body.email", "body.id"]
    assert {v.keyword for v in violations} == {"format", "minimum"}

    message = failure(lambda: assert_matches_schema(responses["user"], SCHEMA))
    assert message.startswith("GET /users/{id} -> 200")
    assert "body.password_hint: value fails 'type'" in message
    assert "s3cr3t-hint" not in message


def test_invalid_schemas_and_loading(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not a valid JSON Schema"):
        schema_violations({}, {"type": "no-such-type"})
    path = tmp_path / "user.schema.yaml"
    path.write_text("type: object\nrequired: [id]\n", encoding="utf-8")
    assert load_schema(path) == {"type": "object", "required": ["id"]}
    with pytest.raises(ContractError, match="cannot load"):
        load_schema(tmp_path / "missing.json")
