"""Test doubles at the HTTP boundary: an Ollama client whose transport is scripted.

These prove client and evaluation *logic*. They are not evidence that any real model works.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from complete_automation.llm_evaluation import (
    EvaluationConfig,
    GenerationConfig,
    JudgeConfig,
    ModelConfig,
    OllamaClient,
)

TARGET = ModelConfig(model="target-model", timeout_seconds=5, generation=GenerationConfig(temperature=0.0))
JUDGE = JudgeConfig(model=ModelConfig(model="judge-model", timeout_seconds=5))


def chat_body(text: str, **extra) -> dict:
    body = {
        "model": "target-model", "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": text}, "done": True, "done_reason": "stop",
        "total_duration": 5_000_000, "prompt_eval_count": 11, "eval_count": 7,
    }
    body.update(extra)
    return body


class Script:
    """Records every request and answers from a function or a queue of responses."""

    def __init__(
        self, handler: Callable[[httpx.Request], httpx.Response] | None = None, *, tags: tuple[str, ...] | None = None
    ) -> None:
        """``tags``: answer ``GET /api/tags`` with these installed models (not counted in ``requests``)."""
        self.tags = tags
        self.tag_requests = 0
        self.requests: list[httpx.Request] = []
        self.handler = handler
        self.queue: list[httpx.Response | Exception] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.tags is not None and request.url.path == "/api/tags":
            self.tag_requests += 1
            return httpx.Response(200, json={"models": [{"name": n, "model": n, "digest": f"digest-{n}"} for n in self.tags]})
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def payloads(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.content]

    @property
    def calls(self) -> int:
        return len(self.requests)


def client_for(script: Script) -> OllamaClient:
    return OllamaClient(transport=httpx.MockTransport(script))


def reply(text: str, **extra) -> httpx.Response:
    return httpx.Response(200, json=chat_body(text, **extra))


def judge_reply(criterion: str, score: float = 4, verdict: str = "pass", rationale: str = "ok", **extra) -> httpx.Response:
    payload = {"criterion": criterion, "score": score, "verdict": verdict, "rationale": rationale, **extra}
    return reply(json.dumps(payload))


def config(**overrides) -> EvaluationConfig:
    return EvaluationConfig(model=TARGET, **overrides)
