"""Ollama client behaviour against a scripted HTTP transport (mocked: no server, no real model)."""

import json

import httpx
import pytest

from complete_automation.llm_evaluation import (
    GenerationConfig,
    MalformedResponseError,
    Message,
    ModelConfig,
    ModelConnectionError,
    ModelHTTPError,
    ModelInvocationError,
    ModelNotFoundError,
    ModelRequest,
    ModelTimeoutError,
)
from complete_automation.llm_evaluation.schemas import RequestStatus
from tests.unit.llm_evaluation.helpers import Script, chat_body, client_for, reply

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]


def request(**overrides) -> ModelRequest:
    config = ModelConfig(model="m1", base_url="http://ollama.test:9999", **overrides.pop("config", {}))
    return ModelRequest(
        messages=(Message(role="system", content="be brief"), Message(role="user", content="hi")),
        config=config, **overrides,
    )


def test_successful_chat_response_is_parsed_and_normalised() -> None:
    script = Script()
    script.queue.append(reply("Hello!", prompt_eval_count=12, eval_count=3))
    response = client_for(script).generate(request())
    assert response.status is RequestStatus.COMPLETED and response.ok
    assert response.text == "Hello!"
    assert (response.provider, response.model) == ("ollama", "m1")
    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (12, 3, "stop")
    assert response.latency_seconds >= 0
    assert response.provider_metadata["total_duration_ns"] == 5_000_000
    assert response.model_identifier is None  # only /api/tags exposes a digest; never invented


def test_endpoint_model_messages_and_generation_options_are_sent() -> None:
    script = Script()
    script.queue.append(reply("x"))
    gen = GenerationConfig(temperature=0.2, top_p=0.9, max_tokens=50, seed=7, stop=("END",), think=False)
    client_for(script).generate(request(config={"generation": gen}, response_format="json"))
    (sent,) = script.requests
    assert str(sent.url) == "http://ollama.test:9999/api/chat"
    body = json.loads(sent.content)
    assert body["model"] == "m1" and body["stream"] is False and body["format"] == "json"
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    assert body["options"] == {"temperature": 0.2, "top_p": 0.9, "num_predict": 50, "seed": 7, "stop": ["END"]}
    assert body["think"] is False


def test_unset_generation_parameters_are_not_sent() -> None:
    script = Script()
    script.queue.append(reply("x"))
    client_for(script).generate(request())
    body = script.payloads()[0]
    assert "options" not in body and "think" not in body and "format" not in body


def test_generate_mode_uses_generate_endpoint_with_system_field() -> None:
    script = Script()
    script.queue.append(httpx.Response(200, json={"response": "ok", "done": True, "done_reason": "stop"}))
    response = client_for(script).generate(request(config={"mode": "generate"}))
    assert str(script.requests[0].url).endswith("/api/generate")
    assert script.payloads()[0]["prompt"] == "hi" and script.payloads()[0]["system"] == "be brief"
    assert response.text == "ok" and response.input_tokens is None and response.output_tokens is None


def test_thinking_content_is_kept_apart_from_the_answer() -> None:
    script = Script()
    script.queue.append(httpx.Response(200, json=chat_body("answer", message={"role": "assistant", "content": "answer", "thinking": "hmm"})))
    response = client_for(script).generate(request())
    assert response.text == "answer" and response.thinking == "hmm"


def test_empty_response_is_returned_as_empty_text_not_an_error() -> None:
    script = Script()
    script.queue.append(reply(""))
    response = client_for(script).generate(request())
    assert response.ok and response.text == ""


def test_missing_optional_metadata_is_unavailable_not_invented() -> None:
    script = Script()
    script.queue.append(httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}, "done": True}))
    response = client_for(script).generate(request())
    assert response.input_tokens is None and response.output_tokens is None and response.finish_reason is None


def test_timeout_is_a_timeout_error() -> None:
    script = Script()
    script.queue.append(httpx.ReadTimeout("slow"))
    with pytest.raises(ModelTimeoutError, match="did not answer within"):
        client_for(script).generate(request())


def test_connection_failure_is_a_clear_connection_error() -> None:
    script = Script()
    script.queue.append(httpx.ConnectError("refused"))
    with pytest.raises(ModelConnectionError, match=r"cannot connect to Ollama at http://ollama.test:9999"):
        client_for(script).generate(request())


def test_unknown_model_is_reported_with_the_model_name() -> None:
    script = Script()
    script.queue.append(httpx.Response(404, json={"error": "model 'm1' not found"}))
    with pytest.raises(ModelNotFoundError, match="m1") as info:
        client_for(script).generate(request())
    assert info.value.status_code == 404


def test_http_error_carries_status_and_server_message() -> None:
    script = Script()
    script.queue.append(httpx.Response(500, json={"error": "out of memory"}))
    with pytest.raises(ModelHTTPError, match="500.*out of memory") as info:
        client_for(script).generate(request())
    assert info.value.status_code == 500


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["list"]),
        httpx.Response(200, json={"done": True}),
        httpx.Response(200, json={"message": {"role": "assistant", "content": 5}, "done": True}),
        httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}}),
    ],
    ids=["not-json", "not-object", "no-message", "non-text-content", "not-done"],
)
def test_malformed_responses_are_rejected(response: httpx.Response) -> None:
    script = Script()
    script.queue.append(response)
    with pytest.raises(MalformedResponseError):
        client_for(script).generate(request())


def test_generation_is_never_retried() -> None:
    script = Script()
    script.queue.extend([httpx.ConnectError("refused"), reply("late")])
    with pytest.raises(ModelInvocationError):
        client_for(script).generate(request())
    assert script.calls == 1


def test_generate_mode_with_several_user_messages_is_refused() -> None:
    bad = ModelRequest(
        messages=(Message(role="user", content="a"), Message(role="user", content="b")),
        config=ModelConfig(model="m1", mode="generate"),
    )
    with pytest.raises(ModelInvocationError, match="exactly one"):
        client_for(Script()).generate(bad)


# ---- streaming


def ndjson(*chunks: dict) -> httpx.Response:
    return httpx.Response(200, content="\n".join(json.dumps(c) for c in chunks).encode())


def piece(text: str, **extra) -> dict:
    return {"message": {"role": "assistant", "content": text}, "done": False, **extra}


def test_streaming_completion_collects_text_and_final_metadata() -> None:
    script = Script()
    script.queue.append(ndjson(
        piece("Hel"), piece("lo"),
        {"message": {"role": "assistant", "content": "!"}, "done": True, "done_reason": "stop", "eval_count": 3, "prompt_eval_count": 9},
    ))
    response = client_for(script).generate(request(config={"stream": True}))
    assert response.status is RequestStatus.COMPLETED
    assert response.text == "Hello!" and response.output_tokens == 3 and response.input_tokens == 9
    assert script.payloads()[0]["stream"] is True


def test_interrupted_stream_is_partial_not_completed() -> None:
    script = Script()
    script.queue.append(ndjson(piece("Hel"), piece("lo")))  # ends without done
    response = client_for(script).generate(request(config={"stream": True}))
    assert response.status is RequestStatus.PARTIAL and not response.ok
    assert response.text == "Hello" and response.error.kind == "stream_incomplete"


def test_stream_error_chunk_after_text_is_partial_with_error() -> None:
    script = Script()
    script.queue.append(ndjson(piece("Hi"), {"error": "model crashed"}))
    response = client_for(script).generate(request(config={"stream": True}))
    assert response.status is RequestStatus.PARTIAL and response.error.message == "model crashed"


def test_stream_error_before_any_text_raises() -> None:
    script = Script()
    script.queue.append(ndjson({"error": "model crashed"}))
    with pytest.raises(ModelInvocationError, match="model crashed"):
        client_for(script).generate(request(config={"stream": True}))


def test_stream_http_error_raises() -> None:
    script = Script()
    script.queue.append(httpx.Response(404, json={"error": "model 'm1' not found"}))
    with pytest.raises(ModelNotFoundError):
        client_for(script).generate(request(config={"stream": True}))


# ---- availability


def tags(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": n, "model": n, "digest": f"digest-{n}"} for n in names]})


def test_check_model_finds_installed_model_and_its_digest() -> None:
    script = Script()
    script.queue.append(tags("qwen3:4b", "other:latest"))
    availability = client_for(script).check_model(ModelConfig(model="qwen3:4b"))
    assert availability.available and availability.identifier == "digest-qwen3:4b"
    assert str(script.requests[0].url) == "http://localhost:11434/api/tags"


def test_check_model_resolves_implicit_latest_tag() -> None:
    script = Script()
    script.queue.append(tags("other:latest"))
    assert client_for(script).check_model(ModelConfig(model="other")).available


def test_check_model_reports_missing_model_and_what_is_installed() -> None:
    script = Script()
    script.queue.append(tags("a:1"))
    availability = client_for(script).check_model(ModelConfig(model="missing"))
    assert not availability.available and "a:1" in availability.detail and "ollama list" in availability.detail


def test_check_model_reports_unreachable_server_without_raising() -> None:
    script = Script()
    script.queue.append(httpx.ConnectError("refused"))
    availability = client_for(script).check_model(ModelConfig(model="m"))
    assert not availability.available and "cannot connect" in availability.detail
