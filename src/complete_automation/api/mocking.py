"""Service virtualization on ``httpx.MockTransport`` for tests without a network.

A mocked test proves the *client* behaves correctly for a given response. It
does not prove a real service sends that response; use contract tests and
integration tests against a running service for that.

    mock = MockApi()
    mock.route("GET", "/users/{user_id}").respond(200, json={"id": 1})
    mock.route("GET", "/flaky").respond_sequence([Reply(503), Reply(200, json={})])
    mock.route("GET", "/slow").fail_with(httpx.ReadTimeout)
    with mock.client() as client:
        assert client.get("/users/{user_id}", path_params={"user_id": 1}).status_code == 200
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from complete_automation.api.client import ApiClient, AsyncApiClient, path_template_regex

_UNSET: Any = object()


@dataclass(frozen=True)
class Reply:
    """One canned response, or a transport error to raise instead."""

    status: int = 200
    json: Any = _UNSET
    text: str | None = None
    content: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    error: type[httpx.TransportError] | None = None

    def build(self, request: httpx.Request) -> httpx.Response:
        if self.error is not None:
            raise self.error("simulated transport failure", request=request)
        kwargs: dict[str, Any] = {"headers": self.headers}
        if self.json is not _UNSET:
            kwargs["json"] = self.json
        elif self.text is not None:
            kwargs["text"] = self.text
        elif self.content is not None:
            kwargs["content"] = self.content
        return httpx.Response(self.status, request=request, **kwargs)


class Route:
    def __init__(self, method: str, template: str) -> None:
        self.method = method.upper()
        self.template = template
        self._regex = path_template_regex(template)
        self._replies: list[Reply] = []
        self._handler: Callable[[httpx.Request], httpx.Response] | None = None
        self.calls: list[httpx.Request] = []

    def matches(self, method: str, path: str) -> bool:
        return method == self.method and self._regex.match(path) is not None

    def respond(self, status: int = 200, *, json: Any = _UNSET, text: str | None = None,
                content: bytes | None = None, headers: dict[str, str] | None = None) -> Route:
        """Always answer with this response (replaces earlier replies)."""
        self._replies = [Reply(status, json, text, content, dict(headers or {}))]
        self._handler = None
        return self

    def respond_sequence(self, replies: Iterable[Reply]) -> Route:
        """Answer with each reply in turn; the last one repeats."""
        self._replies = list(replies)
        if not self._replies:
            raise ValueError("a sequence needs at least one reply")
        self._handler = None
        return self

    def fail_with(self, error: type[httpx.TransportError]) -> Route:
        """Raise this transport error (e.g. ``httpx.ReadTimeout``) on every call."""
        return self.respond_sequence([Reply(error=error)])

    def handle(self, handler: Callable[[httpx.Request], httpx.Response]) -> Route:
        """Compute responses with a function of the request."""
        self._handler = handler
        self._replies = []
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self._handler is not None:
            return self._handler(request)
        reply = self._replies[0] if len(self._replies) == 1 else self._replies.pop(0)
        return reply.build(request)


class MockApi:
    """A table of routes served through ``httpx.MockTransport``."""

    def __init__(self, base_url: str = "https://api.example.test") -> None:
        self.base_url = base_url
        self._base_path = httpx.URL(base_url).raw_path.decode("ascii").split("?", 1)[0].rstrip("/")
        self.routes: list[Route] = []
        self.unmatched: list[httpx.Request] = []

    def route(self, method: str, template: str) -> Route:
        route = Route(method, template)
        self.routes.insert(0, route)  # later definitions win
        return route

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        # Route on the percent-encoded path, as a server does: an encoded "/" stays in its segment.
        path = request.url.raw_path.decode("ascii").split("?", 1)[0]
        if self._base_path and path.startswith(self._base_path):
            path = path[len(self._base_path):] or "/"
        for route in self.routes:
            if route.matches(request.method, path):
                return route(request)
        self.unmatched.append(request)
        return httpx.Response(
            501, json={"error": f"no mock route for {request.method} {path}"}, request=request
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._dispatch)

    def client(self, **options: Any) -> ApiClient:
        return ApiClient(options.pop("base_url", self.base_url), transport=self.transport(), **options)

    def async_client(self, **options: Any) -> AsyncApiClient:
        return AsyncApiClient(options.pop("base_url", self.base_url), transport=self.transport(), **options)

    def assert_no_unmatched(self) -> None:
        if self.unmatched:
            seen = ", ".join(f"{r.method} {r.url.path}" for r in self.unmatched)
            raise AssertionError(f"requests without a mock route: {seen}")
