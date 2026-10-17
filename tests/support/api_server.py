"""A local HTTP service for API integration and example tests (standard library only).

It implements the example contract ``data/api/contracts/users_api.openapi.yaml``
under ``/v1`` and adds test-only endpoints (flaky, slow, redirects, upload,
download, OAuth2 token, ...). It binds to 127.0.0.1 on a free port; nothing
leaves the machine. ``GET /v1/legacy/users/{id}`` deliberately breaks the
contract.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from collections.abc import Iterator
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

VALID_TOKEN = "test-token"
CLIENT_ID, CLIENT_SECRET = "client-1", "secret-1"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _seed() -> dict[int, dict[str, Any]]:
    names = ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Edsger Dijkstra", "Barbara Liskov"]
    return {
        i: {
            "id": i,
            "name": name,
            "email": f"{name.split()[0].lower()}@example.test",
            "role": "admin" if i == 1 else "member",
            "created_at": f"2026-01-0{i}T09:00:00Z",
        }
        for i, name in enumerate(names, start=1)
    }


class ApiService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "lock", threading.Lock()):
            self.users = _seed()
            self.next_id = len(self.users) + 1
            self.counters: dict[str, int] = {}
            self.tokens_issued = 0
            self.requests: list[tuple[str, str, dict[str, str]]] = []

    def count(self, key: str) -> int:
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + 1
            return self.counters[key]


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # silence stderr
        pass

    # -- helpers -----------------------------------------------------------------------

    def _send(self, status: int, body: Any = None, *, headers: dict[str, str] | None = None,
              raw: bytes | None = None, content_type: str = "application/json") -> None:
        payload = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
        self.send_response(status)
        if payload or raw is not None:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _body(self) -> bytes:
        return self._request_body

    def _read_body(self) -> bytes:
        # Always consume the body: an unread body would corrupt the next request on a kept-alive connection.
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {VALID_TOKEN}":
            return True
        self._send(401, {"error": "unauthorized"}, headers={"WWW-Authenticate": "Bearer"})
        return False

    def _route(self) -> None:
        self._request_body = self._read_body()
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        service = self.server.service
        service.requests.append((self.command, self.path, dict(self.headers.items())))
        method = self.command
        if self.server.role == "other":
            return self._send(200, {"path": path, "headers": dict(self.headers.items()), "query": query})
        if not path.startswith("/v1/") and path != "/v1":
            return self._send(404, {"error": "not found"})
        path = path[len("/v1"):] or "/"

        if path == "/health" and method in ("GET", "HEAD"):
            return self._send(200, {"status": "ok"}, headers={"Server-Timing": "app;dur=1.5"})
        if path == "/health" and method == "OPTIONS":
            return self._send(204, headers={"Allow": "GET, HEAD, OPTIONS"})
        if path == "/error":
            return self._send(500, {"error": "internal error"})
        if path == "/malformed":
            return self._send(200, raw=b'{"id": 1, "name": ', content_type="application/json")
        if path == "/text":
            return self._send(200, raw=b"plain text body", content_type="text/plain; charset=utf-8")
        if path == "/download":
            return self._send(200, raw=bytes(range(256)), content_type="application/octet-stream")
        if path == "/slow":
            time.sleep(float(query.get("delay", ["0.5"])[0]))
            return self._send(200, {"slept": True})
        if path in ("/flaky", "/flaky-post"):
            key = f"{method} {query.get('key', ['default'])[0]}"
            attempt = service.count(key)
            failures = int(query.get("fail", ["2"])[0])
            if attempt <= failures:
                headers = {"Retry-After": query["retry_after"][0]} if "retry_after" in query else {}
                return self._send(int(query.get("status", ["503"])[0]), {"error": "try again"}, headers=headers)
            return self._send(200, {"attempts": attempt})
        if path == "/redirect":
            return self._send(int(query.get("status", ["302"])[0]), headers={"Location": query["to"][0]})
        if path == "/echo":
            return self._send(200, {"method": method, "headers": dict(self.headers.items()), "query": query,
                                     "body": self._body().decode("utf-8", errors="replace")})
        if path == "/cookies/set":
            return self._send(200, {"set": True}, headers={"Set-Cookie": "session=abc123; Path=/"})
        if path == "/form" and method == "POST":
            return self._send(200, {k: v[0] for k, v in parse_qs(self._body().decode()).items()})
        if path == "/upload" and method == "POST":
            message = BytesParser(policy=HTTP).parsebytes(
                f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + self._body()
            )
            files = [
                {"field": part.get_param("name", header="content-disposition"), "filename": part.get_filename(),
                 "size": len(part.get_payload(decode=True) or b""), "content_type": part.get_content_type()}
                for part in message.iter_parts()
            ]
            return self._send(200, {"files": files})
        if path == "/token" and method == "POST":
            return self._token()
        if path.startswith("/users") or path.startswith("/legacy/"):
            if not self._authorized():
                return None
            return self._users(method, path, query)
        return self._send(404, {"error": "not found"})

    def _token(self) -> None:
        form = {k: v[0] for k, v in parse_qs(self._body().decode()).items()}
        client_id, secret = form.get("client_id"), form.get("client_secret")
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            client_id, _, secret = base64.b64decode(header[6:]).decode().partition(":")
        if (client_id, secret) != (CLIENT_ID, CLIENT_SECRET):
            return self._send(401, {"error": "invalid_client"})
        if form.get("grant_type") != "client_credentials":
            return self._send(400, {"error": "unsupported_grant_type"})
        with self.server.service.lock:
            self.server.service.tokens_issued += 1
        return self._send(200, {"access_token": VALID_TOKEN, "token_type": "Bearer", "expires_in": 3600,
                                "scope": form.get("scope", "")})

    def _users(self, method: str, path: str, query: dict[str, list[str]]) -> None:
        service = self.server.service
        match = re.fullmatch(r"/users/(\d+)", path)
        legacy = re.fullmatch(r"/legacy/users/(\d+)", path)
        if path == "/users" and method == "GET":
            try:
                page = int(query.get("page", ["1"])[0])
                per_page = int(query.get("per_page", ["2"])[0])
                if page < 1 or not 1 <= per_page <= 50:
                    raise ValueError
            except ValueError:
                return self._send(422, {"error": "invalid pagination",
                                        "details": [{"field": "page", "message": "must be a positive integer"}]})
            users = sorted(service.users.values(), key=lambda u: u["id"])
            total = len(users)
            data = users[(page - 1) * per_page: page * per_page]
            return self._send(200, {"page": page, "per_page": per_page, "total": total,
                                    "total_pages": -(-total // per_page), "data": data})
        if path == "/users" and method == "POST":
            try:
                body = json.loads(self._body() or b"null")
            except ValueError:
                return self._send(422, {"error": "body is not JSON"})
            details = []
            if not isinstance(body, dict):
                body, details = {}, [{"field": "body", "message": "must be an object"}]
            if not isinstance(body.get("name"), str) or not body.get("name"):
                details.append({"field": "name", "message": "is required"})
            if not isinstance(body.get("email"), str) or not _EMAIL.match(body.get("email", "")):
                details.append({"field": "email", "message": "must be an email address"})
            if body.get("role", "member") not in ("admin", "member"):
                details.append({"field": "role", "message": "must be admin or member"})
            if details:
                return self._send(422, {"error": "validation failed", "details": details})
            with service.lock:
                user = {"id": service.next_id, "name": body["name"], "email": body["email"],
                        "role": body.get("role", "member"), "created_at": "2026-10-17T09:00:00Z"}
                service.users[user["id"]] = user
                service.next_id += 1
            return self._send(201, user, headers={"Location": f"/v1/users/{user['id']}"})
        if match and method == "GET":
            user = service.users.get(int(match.group(1)))
            return self._send(200, user) if user else self._send(404, {"error": "user not found"})
        if match and method == "DELETE":
            with service.lock:
                removed = service.users.pop(int(match.group(1)), None)
            return self._send(204) if removed else self._send(404, {"error": "user not found"})
        if legacy and method == "GET":
            user = service.users.get(int(legacy.group(1)))
            if not user:
                return self._send(404, {"error": "user not found"})
            # Deliberate contract break: id as a string, undocumented extra field.
            return self._send(200, {**user, "id": str(user["id"]), "nickname": "legacy"})
        return self._send(405, {"error": "method not allowed"})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _route


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: ApiService, role: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.service = service
        self.role = role

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # clients that time out and disconnect are expected in these tests


class RunningService:
    """A started server; ``base_url`` ends in ``/v1`` for the main service."""

    def __init__(self, role: str = "main") -> None:
        self.service = ApiService()
        self._server = _Server(self.service, role)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.origin = f"http://{host}:{port}"
        self.base_url = f"{self.origin}/v1" if role == "main" else self.origin

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture(scope="session")
def _api_services() -> Iterator[tuple[RunningService, RunningService]]:
    main, other = RunningService("main"), RunningService("other")
    try:
        yield main, other
    finally:
        main.stop()
        other.stop()


@pytest.fixture
def local_api(_api_services: tuple[RunningService, RunningService]) -> RunningService:
    """The example Users API on 127.0.0.1, reset to its seed data for each test."""
    main, _ = _api_services
    main.service.reset()
    return main


@pytest.fixture
def other_origin(_api_services: tuple[RunningService, RunningService]) -> RunningService:
    """A second local server (different port, so a different origin) that echoes requests."""
    return _api_services[1]
