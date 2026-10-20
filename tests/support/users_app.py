"""A minimal users application for cross-module workflows (standard library only).

It is the system under test for ``tests/integration/cross_module``: one SQLite file
holds the data, a JSON API under ``/api`` and a server-rendered HTML page under
``/users`` read and change it. The API, database and UI modules each see the same
records through their own channel, which is what the workflows verify. It binds to
127.0.0.1 on a free port; nothing leaves the machine.

Schema (``users``): ``id`` integer primary key, ``name`` non-empty text, ``email``
unique text, ``role`` ``admin`` or ``member``, ``created_at`` ISO 8601 UTC text.

API:

=======================  ====================================================================
``GET /api/health``      ``{"status": "ok"}``
``GET /api/users``       ``{"data": [record, ...]}``, ordered by ``id``
``POST /api/users``      ``{name, email, role?}`` -> 201 and the record; 422 with ``details``
                         for invalid input; 409 for a duplicate email (nothing is written)
``GET /api/users/{id}``  200 and the record, or 404
``PATCH /api/users/{id}`` ``{name?, role?}`` -> 200 and the record; 404; 422
``DELETE /api/users/{id}`` 204, or 404
=======================  ====================================================================

HTML: ``GET /users`` lists every user (one ``data-testid="user-row"`` per record, with
``Promote`` and ``Delete`` buttons that post to ``/users/{id}/promote`` and
``/users/{id}/delete`` and redirect back to the list).

It is a test fixture, not a production application: no authentication, no
pagination, one writer at a time (SQLite).
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL CHECK (length(name) > 0),
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'member')),
    created_at TEXT NOT NULL
)
"""
ROLES = ("admin", "member")
FIXED_NOW = "2026-10-20T09:00:00Z"
"""The ``created_at`` the test fixtures give every record (``now=lambda: FIXED_NOW``)."""
SEED_USERS = (
    ("Seed Admin", "seed.admin@example.test", "admin"),
    ("Seed Member", "seed.member@example.test", "member"),
)
"""Unrelated records the test fixtures create first; a workflow must leave them untouched."""
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_COLUMNS = "id, name, email, role, created_at"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class UsersStore:
    """The application's own data access: one short-lived connection per operation."""

    def __init__(self, path: Path, *, now: Callable[[], str] = _utc_now) -> None:
        self.path = path
        self.now = now
        with closing(self._connect()) as connection, connection:
            connection.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def list(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            return [dict(row) for row in connection.execute(f"SELECT {_COLUMNS} FROM users ORDER BY id")]

    def get(self, user_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(f"SELECT {_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def create(self, name: str, email: str, role: str) -> dict[str, Any]:
        """Insert one user; raises :class:`sqlite3.IntegrityError` for a duplicate email."""
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO users (name, email, role, created_at) VALUES (?, ?, ?, ?)",
                (name, email, role, self.now()),
            )
            user_id = cursor.lastrowid
        created = self.get(int(user_id or 0))
        assert created is not None
        return created

    def update(self, user_id: int, changes: dict[str, str]) -> dict[str, Any] | None:
        if changes:
            assignments = ", ".join(f"{column} = ?" for column in changes)  # columns are validated by the caller
            with closing(self._connect()) as connection, connection:
                connection.execute(f"UPDATE users SET {assignments} WHERE id = ?", (*changes.values(), user_id))
        return self.get(user_id)

    def delete(self, user_id: int) -> bool:
        with closing(self._connect()) as connection, connection:
            return connection.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount == 1


def _validate(body: Any, *, partial: bool) -> tuple[dict[str, str], list[dict[str, str]]]:
    if not isinstance(body, dict):
        return {}, [{"field": "body", "message": "must be an object"}]
    values: dict[str, str] = {}
    details: list[dict[str, str]] = []
    unknown = sorted(set(body) - ({"name", "role"} if partial else {"name", "email", "role"}))
    details += [{"field": field, "message": "is not allowed"} for field in unknown]
    if "name" in body or not partial:
        if isinstance(body.get("name"), str) and body["name"].strip():
            values["name"] = body["name"].strip()
        else:
            details.append({"field": "name", "message": "is required"})
    if not partial:
        if isinstance(body.get("email"), str) and _EMAIL.match(body["email"]):
            values["email"] = body["email"].lower()
        else:
            details.append({"field": "email", "message": "must be an email address"})
    if "role" in body or not partial:
        role = body.get("role", "member")
        if role in ROLES:
            values["role"] = role
        else:
            details.append({"field": "role", "message": "must be admin or member"})
    return values, details


def _page(users: list[dict[str, Any]], notice: str | None) -> str:
    rows = "\n".join(
        f'<tr data-testid="user-row" data-user-id="{user["id"]}">'
        f'<td data-testid="user-name">{html.escape(user["name"])}</td>'
        f'<td data-testid="user-email">{html.escape(user["email"])}</td>'
        f'<td data-testid="user-role">{html.escape(user["role"])}</td>'
        f'<td><form method="post" action="/users/{user["id"]}/promote">'
        f'<button type="submit" aria-label="Promote {html.escape(user["email"])}">Promote</button></form>'
        f'<form method="post" action="/users/{user["id"]}/delete">'
        f'<button type="submit" aria-label="Delete {html.escape(user["email"])}">Delete</button></form></td></tr>'
        for user in users
    )
    banner = f'<p role="status" data-testid="notice">{html.escape(notice)}</p>' if notice else ""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Users</title></head><body>"
        f"<h1>Users</h1>{banner}"
        f'<p data-testid="user-count">{len(users)} users</p>'
        "<table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Actions</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature from the base class
        return

    def _send(self, status: int, body: Any = None, *, headers: dict[str, str] | None = None,
              content_type: str = "application/json") -> None:
        if isinstance(body, str):
            payload = body.encode("utf-8")
        else:
            payload = b"" if body is None else json.dumps(body).encode("utf-8")
        self.send_response(status)
        if payload:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def _json_body(self) -> tuple[bool, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return True, json.loads(raw or b"null")
        except ValueError:
            return False, None

    def _route(self) -> None:
        store = self.server.store
        parts = urlsplit(self.path)
        path, method = parts.path, self.command
        api_user = re.fullmatch(r"/api/users/(\d+)", path)
        page_action = re.fullmatch(r"/users/(\d+)/(promote|delete)", path)

        if path == "/api/health" and method == "GET":
            return self._send(200, {"status": "ok"})
        if path == "/api/users" and method == "GET":
            return self._send(200, {"data": store.list()})
        if path == "/api/users" and method == "POST":
            ok, body = self._json_body()
            if not ok:
                return self._send(422, {"error": "body is not JSON"})
            values, details = _validate(body, partial=False)
            if details:
                return self._send(422, {"error": "validation failed", "details": details})
            try:
                user = store.create(values["name"], values["email"], values["role"])
            except sqlite3.IntegrityError:
                return self._send(409, {"error": "email already exists"})
            return self._send(201, user, headers={"Location": f"/api/users/{user['id']}"})
        if api_user and method == "GET":
            user = store.get(int(api_user.group(1)))
            return self._send(200, user) if user else self._send(404, {"error": "user not found"})
        if api_user and method == "PATCH":
            ok, body = self._json_body()
            values, details = _validate(body, partial=True) if ok else ({}, [{"field": "body", "message": "not JSON"}])
            if details:
                return self._send(422, {"error": "validation failed", "details": details})
            user = store.update(int(api_user.group(1)), values)
            return self._send(200, user) if user else self._send(404, {"error": "user not found"})
        if api_user and method == "DELETE":
            return self._send(204) if store.delete(int(api_user.group(1))) else self._send(404, {"error": "user not found"})
        if path == "/users" and method == "GET":
            notice = {"promoted": "User promoted", "deleted": "User deleted"}.get(parts.query.partition("=")[0])
            return self._send(200, _page(store.list(), notice), content_type="text/html; charset=utf-8")
        if page_action and method == "POST":
            self._json_body()  # consume the form body; the action is in the path
            user_id, action = int(page_action.group(1)), page_action.group(2)
            done = store.delete(user_id) if action == "delete" else store.update(user_id, {"role": "admin"}) is not None
            if not done:
                return self._send(404, "<p>User not found</p>", content_type="text/html; charset=utf-8")
            return self._send(303, headers={"Location": f"/users?{action}d={user_id}"})
        return self._send(404, {"error": "not found"})

    do_GET = do_POST = do_PATCH = do_DELETE = _route


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, store: UsersStore) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.store = store

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # a client that disconnects early is not a test failure


class UsersApp:
    """A started application: ``api_url`` for the API client, ``url`` for the browser, ``db_path`` for the database."""

    def __init__(self, db_path: Path, *, now: Callable[[], str] = _utc_now) -> None:
        self.db_path = db_path
        self.store = UsersStore(db_path, now=now)
        self._server = _Server(self.store)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.api_url = f"{self.url}/api"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
