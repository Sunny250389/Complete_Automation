"""Workflow A: what the API reports is what the database holds.

Each test drives the local users application (tests/support/users_app.py) through
the API module and verifies the persisted state through the database module, on a
read-only connection to the application's own SQLite file. Every test has a fresh
database with two unrelated seed users that must survive untouched.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from complete_automation.api import ApiClient
from complete_automation.api.assertions import (
    assert_error_response,
    assert_header,
    assert_json_keys,
    assert_status,
)
from complete_automation.core.datasets import DataFactory
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.database import ReconciliationConfig, UnsafeOperationError, reconcile
from complete_automation.database.session import DatabaseSession
from complete_automation.database.validations import (
    assert_record,
    assert_row_absent,
    assert_row_count,
)
from tests.support.users_app import FIXED_NOW, SEED_USERS, UsersApp

pytestmark = [
    pytest.mark.api,
    pytest.mark.database,
    pytest.mark.integration,
    pytest.mark.cross_module,
    pytest.mark.cauto_module("cross_module"),
]

USER_FIELDS = ["id", "name", "email", "role", "created_at"]


@pytest.fixture
def api(api_client_factory: Callable[..., ApiClient], users_app: UsersApp) -> ApiClient:
    return api_client_factory(base_url=users_app.api_url)


@pytest.fixture
def data() -> DataFactory:
    return DataFactory(seed=20261020)


def new_user(data: DataFactory, role: str = "member") -> dict[str, str]:
    return {"name": f"Workflow {data.sequence('user')}", "email": data.email(), "role": role}


def assert_seed_users_untouched(db: DatabaseSession) -> None:
    with db.transaction():
        for name, email, role in SEED_USERS:
            assert_record(db, "users", {"email": email}, {"name": name, "role": role})


def test_a_created_user_is_persisted_exactly_as_the_api_reports_it(
    api: ApiClient, users_db_session: DatabaseSession, data: DataFactory, result_recorder: ResultRecorder
) -> None:
    payload = new_user(data, role="admin")

    created = api.post("/users", json=payload)

    assert_status(created, 201)
    assert_json_keys(created, USER_FIELDS, allow_extra=False)
    user = created.json()
    assert_header(created, "Location", f"/api/users/{user['id']}")
    assert {key: user[key] for key in payload} == payload
    assert user["created_at"] == FIXED_NOW
    fetched = api.get(f"/users/{user['id']}")
    assert_status(fetched, 200)
    assert fetched.json() == user

    with users_db_session.transaction():
        row = assert_record(users_db_session, "users", {"id": user["id"]}, {k: v for k, v in user.items() if k != "id"})
        assert_row_count(users_db_session, "users", len(SEED_USERS) + 1)
    assert row == user
    assert_seed_users_untouched(users_db_session)
    result_recorder.add_metadata(workflow="api_database", user_id=user["id"])


def test_the_api_listing_reconciles_with_the_table(
    api: ApiClient, users_db_session: DatabaseSession, data: DataFactory
) -> None:
    for role in ("member", "admin", "member"):
        assert_status(api.post("/users", json=new_user(data, role=role)), 201)

    listed = api.get("/users")
    assert_status(listed, 200)
    with users_db_session.transaction():
        rows = users_db_session.fetch_all("SELECT id, name, email, role, created_at FROM users ORDER BY id")

    result = reconcile(listed.json()["data"], rows, ReconciliationConfig(key=("id",)))
    result.assert_matches()
    assert result.matched == len(SEED_USERS) + 3


def test_rejected_requests_change_nothing_in_the_database(
    api: ApiClient, users_db_session: DatabaseSession, data: DataFactory
) -> None:
    taken = new_user(data)
    assert_status(api.post("/users", json=taken), 201)

    invalid = api.post("/users", json={"name": "", "email": "not-an-email", "role": "owner"})
    duplicate = api.post("/users", json={**new_user(data), "email": taken["email"]})

    assert_error_response(invalid, 422, fields=("error", "details"))
    assert {detail["field"] for detail in invalid.json()["details"]} == {"name", "email", "role"}
    assert_error_response(duplicate, 409)
    with users_db_session.transaction():
        assert_row_count(users_db_session, "users", len(SEED_USERS) + 1)
        assert_row_absent(users_db_session, "users", {"email": "not-an-email"})
        assert_row_count(users_db_session, "users", 1, {"email": taken["email"]})
        assert_record(users_db_session, "users", {"email": taken["email"]}, {"name": taken["name"]})
    assert_seed_users_untouched(users_db_session)


def test_updates_and_deletes_through_the_api_reach_the_database(
    api: ApiClient, users_db_session: DatabaseSession, data: DataFactory
) -> None:
    user = api.post("/users", json=new_user(data)).json()

    promoted = api.patch(f"/users/{user['id']}", json={"role": "admin"})
    assert_status(promoted, 200)
    assert promoted.json()["role"] == "admin"
    with users_db_session.transaction():
        assert_record(users_db_session, "users", {"id": user["id"]}, {"role": "admin", "email": user["email"]})

    assert_status(api.delete(f"/users/{user['id']}"), 204)
    assert_error_response(api.get(f"/users/{user['id']}"), 404)
    assert_error_response(api.delete(f"/users/{user['id']}"), 404)
    with users_db_session.transaction():
        assert_row_absent(users_db_session, "users", {"id": user["id"]})
        assert_row_count(users_db_session, "users", len(SEED_USERS))
    assert_seed_users_untouched(users_db_session)


def test_the_database_connection_used_for_verification_cannot_write(users_db_session: DatabaseSession) -> None:
    with pytest.raises(UnsafeOperationError):
        users_db_session.execute("DELETE FROM users")
    assert_seed_users_untouched(users_db_session)
