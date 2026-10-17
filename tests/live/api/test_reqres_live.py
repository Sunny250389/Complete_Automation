"""Opt-in checks against the public reqres.in demo API (UNVERIFIED in this repository).

These tests reach the internet, so they never run by default: ``api_client``
skips them unless ``modules.api.base_url`` is set. Enable with the ``reqres``
environment and your own key (never commit it)::

    CAUTO_ENVIRONMENT=reqres CAUTO_MODULES__API__AUTH__VALUE=<your key> \
        uv run pytest tests/live/api -m live

The expected payloads follow reqres.in's published examples; the service may
change them without notice, and a failure here says nothing about this framework.
"""

import pytest

from complete_automation.api.assertions import (
    assert_error_response,
    assert_field,
    assert_pagination,
    assert_status,
)

pytestmark = [pytest.mark.api, pytest.mark.live, pytest.mark.cauto_module("api")]


def test_list_users_page(api_client) -> None:
    response = api_client.get("/users", params={"page": 2})

    assert_status(response, 200)
    assert_pagination(response)
    assert_field(response, "data[0].email", pattern=r"[^@]+@[^@]+")


def test_unknown_user_is_404(api_client) -> None:
    assert_status(api_client.get("/users/{user_id}", path_params={"user_id": 23}), 404)


def test_login_without_password_is_rejected(api_client) -> None:
    response = api_client.post("/login", json={"email": "peter@klaven"})

    assert_error_response(response, 400, fields=("error",))
