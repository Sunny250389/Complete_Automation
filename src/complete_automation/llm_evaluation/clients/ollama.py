"""Ollama client for the documented HTTP API (``/api/chat``, ``/api/generate``, ``/api/tags``).

Every call goes to the ``base_url`` of the request's :class:`ModelConfig`; nothing else is contacted,
nothing is pulled or downloaded, and nothing is retried. Optional response fields (token counts,
``done_reason``, durations) are read defensively: when Ollama omits one, the record says so with
``None`` instead of a guess.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from complete_automation.llm_evaluation.errors import (
    MalformedResponseError,
    ModelConnectionError,
    ModelHTTPError,
    ModelInvocationError,
    ModelNotFoundError,
    ModelTimeoutError,
)
from complete_automation.llm_evaluation.schemas import (
    ErrorInfo,
    GenerationConfig,
    ModelConfig,
    ModelRequest,
    ModelResponse,
    RequestStatus,
)

_ERROR_BODY_LIMIT = 300
_DURATION_FIELDS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")


@dataclass(frozen=True)
class ModelAvailability:
    available: bool
    identifier: str | None = None
    detail: str = ""
    installed: tuple[str, ...] = ()


class OllamaClient:
    """Synchronous Ollama client. ``transport`` lets tests replace the HTTP boundary."""

    provider = "ollama"

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport
        self._clients: dict[str, httpx.Client] = {}

    # -- lifecycle -------------------------------------------------------------------------

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _http(self, base_url: str) -> httpx.Client:
        key = base_url.rstrip("/")
        if key not in self._clients:
            self._clients[key] = httpx.Client(base_url=key, transport=self._transport)
        return self._clients[key]

    # -- availability ----------------------------------------------------------------------

    def list_models(self, base_url: str, *, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
        """Installed models as reported by ``GET /api/tags``."""
        started = time.perf_counter()
        try:
            reply = self._http(base_url).get("/api/tags", timeout=timeout_seconds)
        except httpx.TimeoutException as error:
            raise ModelTimeoutError(f"Ollama at {base_url} did not answer within {timeout_seconds}s") from error
        except httpx.TransportError as error:
            raise ModelConnectionError(_connection_message(base_url, error)) from error
        _raise_for_status(reply, started)
        body = _json_object(reply)
        models = body.get("models")
        if not isinstance(models, list):
            raise MalformedResponseError("Ollama /api/tags response has no 'models' list")
        return [m for m in models if isinstance(m, dict)]

    def check_model(self, config: ModelConfig, *, timeout_seconds: float = 10.0) -> ModelAvailability:
        """Is the server reachable and is the configured model installed? Never raises for either."""
        try:
            models = self.list_models(config.base_url, timeout_seconds=timeout_seconds)
        except ModelInvocationError as error:
            return ModelAvailability(False, detail=str(error))
        names = tuple(str(m.get("name") or m.get("model") or "") for m in models)
        wanted = config.model if ":" in config.model else f"{config.model}:latest"
        for entry in models:
            if wanted in (entry.get("name"), entry.get("model")) or config.model == entry.get("name"):
                digest = entry.get("digest")
                return ModelAvailability(True, digest if isinstance(digest, str) else None, installed=names)
        return ModelAvailability(
            False,
            detail=f"model {config.model!r} is not installed at {config.base_url} "
            f"(installed: {', '.join(names) or 'none'}); run `ollama list`",
            installed=names,
        )

    def identify(self, config: ModelConfig) -> str | None:
        """The model's digest from ``/api/tags``; ``None`` if unavailable (never raises)."""
        return self.check_model(config).identifier

    # -- generation ------------------------------------------------------------------------

    def generate(self, request: ModelRequest) -> ModelResponse:
        config = request.config
        endpoint, payload = self._payload(request)
        started = time.perf_counter()
        try:
            if config.stream:
                return self._stream(request, endpoint, payload, started)
            reply = self._http(config.base_url).post(
                endpoint, json=payload, timeout=config.timeout_seconds
            )
        except httpx.TimeoutException as error:
            raise ModelTimeoutError(
                f"Ollama did not answer within {config.timeout_seconds}s ({config.model})"
            ) from error
        except httpx.TransportError as error:
            raise ModelConnectionError(_connection_message(config.base_url, error)) from error
        latency = time.perf_counter() - started
        _raise_for_status(reply, started, model=config.model)
        body = _json_object(reply)
        text, thinking = _extract_text(body, config.mode)
        if body.get("done") is not True:
            raise MalformedResponseError("Ollama response is not marked done (non-streaming call)")
        return self._response(request, RequestStatus.COMPLETED, text, thinking, body, latency)

    def _payload(self, request: ModelRequest) -> tuple[str, dict[str, Any]]:
        config = request.config
        payload: dict[str, Any] = {"model": config.model, "stream": config.stream}
        options = _options(config.generation)
        if options:
            payload["options"] = options
        if config.generation.think is not None:
            payload["think"] = config.generation.think
        if request.response_format is not None:
            payload["format"] = request.response_format
        if config.mode == "chat":
            payload["messages"] = [m.model_dump() for m in request.messages]
            return "/api/chat", payload
        system = [m.content for m in request.messages if m.role == "system"]
        users = [m.content for m in request.messages if m.role != "system"]
        if len(users) != 1:
            raise ModelInvocationError("generate mode takes exactly one non-system message; use chat mode")
        payload["prompt"] = users[0]
        if system:
            payload["system"] = "\n\n".join(system)
        return "/api/generate", payload

    def _stream(
        self, request: ModelRequest, endpoint: str, payload: dict[str, Any], started: float
    ) -> ModelResponse:
        config = request.config
        text: list[str] = []
        thinking: list[str] = []
        final: dict[str, Any] | None = None
        problem: ErrorInfo | None = None
        try:
            with self._http(config.base_url).stream(
                "POST", endpoint, json=payload, timeout=config.timeout_seconds
            ) as reply:
                if reply.status_code >= 400:
                    reply.read()
                    _raise_for_status(reply, started, model=config.model)
                for chunk in _ndjson(reply):
                    if "error" in chunk:
                        problem = ErrorInfo(kind="stream_error", message=_clip(str(chunk["error"])))
                        break
                    piece, think_piece = _extract_text(chunk, config.mode, streaming=True)
                    text.append(piece)
                    if think_piece:
                        thinking.append(think_piece)
                    if chunk.get("done") is True:
                        final = chunk
                        break
                    if time.perf_counter() - started > config.timeout_seconds:
                        problem = ErrorInfo(
                            kind="timeout", message=f"stream exceeded {config.timeout_seconds}s"
                        )
                        break
        except httpx.TimeoutException as error:
            if not any(text):
                raise ModelTimeoutError(
                    f"Ollama did not answer within {config.timeout_seconds}s ({config.model})"
                ) from error
            problem = ErrorInfo(kind="timeout", message="stream stalled")
        except httpx.TransportError as error:
            if not any(text):
                raise ModelConnectionError(_connection_message(config.base_url, error)) from error
            problem = ErrorInfo(kind="stream_interrupted", message=type(error).__name__)
        latency = time.perf_counter() - started
        if final is not None and problem is None:
            return self._response(
                request, RequestStatus.COMPLETED, "".join(text), "".join(thinking) or None, final, latency
            )
        if problem is None:
            problem = ErrorInfo(kind="stream_incomplete", message="stream ended without a final 'done' message")
        if not any(text) and problem.kind == "stream_error":
            raise ModelInvocationError(f"Ollama reported an error: {problem.message}")
        return self._response(
            request, RequestStatus.PARTIAL, "".join(text), "".join(thinking) or None, final or {},
            latency, error=problem,
        )

    def _response(
        self,
        request: ModelRequest,
        status: RequestStatus,
        text: str,
        thinking: str | None,
        body: dict[str, Any],
        latency: float,
        *,
        error: ErrorInfo | None = None,
    ) -> ModelResponse:
        metadata: dict[str, Any] = {}
        for name in _DURATION_FIELDS:
            if isinstance(body.get(name), int):
                metadata[f"{name}_ns"] = body[name]
        if isinstance(body.get("model"), str):
            metadata["server_model"] = body["model"]
        if isinstance(body.get("created_at"), str):
            metadata["created_at"] = body["created_at"]
        return ModelResponse(
            provider=self.provider,
            model=request.config.model,
            request=request,
            status=status,
            text=text,
            thinking=thinking,
            finish_reason=body["done_reason"] if isinstance(body.get("done_reason"), str) else None,
            input_tokens=_count(body.get("prompt_eval_count")),
            output_tokens=_count(body.get("eval_count")),
            latency_seconds=latency,
            error=error,
            provider_metadata=metadata,
        )


def _options(generation: GenerationConfig) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if generation.temperature is not None:
        options["temperature"] = generation.temperature
    if generation.top_p is not None:
        options["top_p"] = generation.top_p
    if generation.max_tokens is not None:
        options["num_predict"] = generation.max_tokens
    if generation.seed is not None:
        options["seed"] = generation.seed
    if generation.stop:
        options["stop"] = list(generation.stop)
    return options


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _clip(text: str) -> str:
    return text if len(text) <= _ERROR_BODY_LIMIT else text[:_ERROR_BODY_LIMIT] + "..."


def _connection_message(base_url: str, error: Exception) -> str:
    return (
        f"cannot connect to Ollama at {base_url} ({type(error).__name__}); "
        "is `ollama serve` running and is the base_url right?"
    )


def _raise_for_status(reply: httpx.Response, started: float, *, model: str | None = None) -> None:
    if reply.status_code < 400:
        return
    detail = ""
    try:
        parsed = json.loads(reply.text)
        if isinstance(parsed, dict) and "error" in parsed:
            detail = str(parsed["error"])
    except ValueError:
        detail = reply.text
    message = f"Ollama returned HTTP {reply.status_code}: {_clip(detail)}".rstrip(": ")
    if reply.status_code == 404 and model is not None:
        raise ModelNotFoundError(f"{message} (model {model!r}; see `ollama list`)", reply.status_code)
    raise ModelHTTPError(message, reply.status_code)


def _json_object(reply: httpx.Response) -> dict[str, Any]:
    try:
        body = json.loads(reply.text)
    except ValueError:
        raise MalformedResponseError("Ollama response is not valid JSON") from None
    if not isinstance(body, dict):
        raise MalformedResponseError("Ollama response is not a JSON object")
    return body


def _ndjson(reply: httpx.Response) -> Iterator[dict[str, Any]]:
    for line in reply.iter_lines():
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except ValueError:
            raise MalformedResponseError("Ollama stream contained a line that is not JSON") from None
        if not isinstance(chunk, dict):
            raise MalformedResponseError("Ollama stream contained a non-object line")
        yield chunk


def _extract_text(body: dict[str, Any], mode: str, *, streaming: bool = False) -> tuple[str, str | None]:
    if mode == "chat":
        message = body.get("message")
        if not isinstance(message, dict):
            if streaming and body.get("done") is True:
                return "", None
            raise MalformedResponseError("Ollama chat response has no 'message' object")
        content, thinking = message.get("content"), message.get("thinking")
    else:
        content, thinking = body.get("response"), body.get("thinking")
    if not isinstance(content, str):
        raise MalformedResponseError(f"Ollama {mode} response has no text content")
    return content, thinking if isinstance(thinking, str) and thinking else None
