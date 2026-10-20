"""Workflow B: one record through the API, the database and the browser.

A user created through the API is verified in the database, found on the
application's page, promoted and deleted with the page's own buttons, and each
change is verified again through the API and the database. Needs the "ui" extra and
an installed browser (``playwright install chromium``).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("playwright")

from playwright.sync_api import Locator, Page  # noqa: E402

from complete_automation.api import ApiClient  # noqa: E402
from complete_automation.api.assertions import assert_error_response, assert_status  # noqa: E402
from complete_automation.core.datasets import DataFactory  # noqa: E402
from complete_automation.core.pytest_plugin import ResultRecorder  # noqa: E402
from complete_automation.database.session import DatabaseSession  # noqa: E402
from complete_automation.database.validations import assert_record, assert_row_absent, assert_row_count  # noqa: E402
from complete_automation.ui.assertions import assert_count, assert_text, assert_title, assert_visible  # noqa: E402
from complete_automation.ui.pages import BasePage  # noqa: E402
from tests.support.users_app import SEED_USERS, UsersApp  # noqa: E402

pytestmark = [
    pytest.mark.api,
    pytest.mark.database,
    pytest.mark.ui,
    pytest.mark.integration,
    pytest.mark.cross_module,
    pytest.mark.cauto_module("cross_module"),
]


class UsersPage(BasePage):
    """The application's user list (``/users``)."""

    path = "/users"

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        self.heading = self.by_role("heading", "Users")
        self.rows = self.by_test_id("user-row")
        self.count = self.by_test_id("user-count")
        self.notice = self.by_test_id("notice")

    def row_for(self, email: str) -> Locator:
        return self.rows.filter(has=self.by_test_id("user-email").filter(has_text=email))

    def promote(self, email: str) -> None:
        self.click(self.by_role("button", f"Promote {email}", exact=True))

    def delete(self, email: str) -> None:
        self.click(self.by_role("button", f"Delete {email}", exact=True))


@pytest.fixture
def api(api_client_factory: Callable[..., ApiClient], users_app: UsersApp) -> ApiClient:
    return api_client_factory(base_url=users_app.api_url)


def test_a_user_created_through_the_api_is_promoted_and_deleted_in_the_browser(
    api: ApiClient,
    users_db_session: DatabaseSession,
    ui_page: Page,
    users_app: UsersApp,
    result_recorder: ResultRecorder,
) -> None:
    data = DataFactory(seed=20261021)
    payload = {"name": f"Workflow {data.sequence('ui-user')}", "email": data.email(), "role": "member"}
    db = users_db_session

    # 1-2. Create through the API, verify in the database.
    created = api.post("/users", json=payload)
    assert_status(created, 201)
    user = created.json()
    with db.transaction():
        assert_record(db, "users", {"id": user["id"]}, payload)

    # 3-4. The browser shows the record with the persisted values.
    page = UsersPage(ui_page, users_app.url)
    page.open()
    assert_title(ui_page, "Users")
    assert_text(page.count, f"{len(SEED_USERS) + 1} users")
    row = page.row_for(payload["email"])
    assert_count(row, 1)
    assert_text(row.get_by_test_id("user-name"), payload["name"])
    assert_text(row.get_by_test_id("user-role"), "member")

    # 5-6. Promote through the page; the API and the database agree.
    page.promote(payload["email"])
    assert_visible(page.notice)
    assert_text(page.notice, "User promoted")
    assert_text(page.row_for(payload["email"]).get_by_test_id("user-role"), "admin")
    promoted = api.get(f"/users/{user['id']}")
    assert_status(promoted, 200)
    assert promoted.json() == {**user, "role": "admin"}
    with db.transaction():
        assert_record(db, "users", {"id": user["id"]}, {"role": "admin"})

    # 5-6. Delete through the page; the record is gone everywhere, the seed users are not.
    page.delete(payload["email"])
    assert_text(page.notice, "User deleted")
    assert_count(page.row_for(payload["email"]), 0)
    assert_text(page.count, f"{len(SEED_USERS)} users")
    assert_error_response(api.get(f"/users/{user['id']}"), 404)
    with db.transaction():
        assert_row_absent(db, "users", {"id": user["id"]})
        assert_row_count(db, "users", len(SEED_USERS))
        for name, email, role in SEED_USERS:
            assert_record(db, "users", {"email": email}, {"name": name, "role": role})
    result_recorder.add_metadata(workflow="api_database_ui", user_id=user["id"])


def test_a_user_deleted_through_the_api_disappears_from_the_page(
    api: ApiClient, users_db_session: DatabaseSession, ui_page: Page, users_app: UsersApp
) -> None:
    data = DataFactory(seed=20261022)
    email = data.email()
    user = api.post("/users", json={"name": "Short-lived user", "email": email}).json()
    page = UsersPage(ui_page, users_app.url)
    page.open()
    assert_count(page.row_for(email), 1)

    assert_status(api.delete(f"/users/{user['id']}"), 204)
    with users_db_session.transaction():
        assert_row_absent(users_db_session, "users", {"id": user["id"]})
    page.open()

    assert_count(page.row_for(email), 0)
    assert_count(page.rows, len(SEED_USERS))
