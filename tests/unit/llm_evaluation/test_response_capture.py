"""The call -> capture -> evaluate flow: the real (here: scripted-transport) response is what gets evaluated."""

import json

import httpx
import pytest

from complete_automation.llm_evaluation import (
    CheckSpec,
    EvaluationRunner,
    GenerationConfig,
    PromptTemplate,
    RAG_TEMPLATE,
    RecordStatus,
    TestCase,
)
from complete_automation.llm_evaluation.schemas import Outcome, RequestStatus
from tests.unit.llm_evaluation.helpers import TARGET, Script, client_for, config, reply

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]


def case(**fields) -> TestCase:
    base = dict(case_id="c1", prompt="Capital of France?", reference="Paris is the capital of France.",
                checks=(CheckSpec(type="contains", value="Paris"),))
    return TestCase(**{**base, **fields})


def runner(script: Script, **config_fields) -> EvaluationRunner:
    return EvaluationRunner(client_for(script), config(**config_fields))


def test_the_evaluated_text_is_what_the_model_returned_not_the_reference() -> None:
    script = Script(tags=("target-model",))
    script.queue.append(reply("Rome, obviously."))
    record = runner(script).run_case(case())
    assert record.response.text == "Rome, obviously."  # not the reference answer
    assert record.status is RecordStatus.FAILED and "Paris" in record.reasons[0]
    assert script.calls == 1


def test_response_metadata_is_captured_and_identifier_comes_from_the_server() -> None:
    script = Script(tags=("target-model",))
    script.queue.append(reply("Paris", prompt_eval_count=5, eval_count=2))
    response = runner(script).invoke(case())
    assert (response.provider, response.model, response.model_identifier) == ("ollama", "target-model", "digest-target-model")
    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (5, 2, "stop")
    assert response.timestamp.tzinfo is not None and response.latency_seconds >= 0
    assert response.request.messages[-1].content == "Capital of France?"


def test_model_identifier_is_unavailable_when_the_lookup_fails() -> None:
    script = Script(lambda request: httpx.Response(500, json={"error": "x"}) if request.url.path == "/api/tags" else reply("Paris"))
    response = EvaluationRunner(client_for(script), config()).invoke(case())
    assert response.model_identifier is None and response.text == "Paris"


def test_evaluating_a_captured_response_again_does_not_call_the_model() -> None:
    script = Script(tags=("target-model",))
    script.queue.append(reply("Paris, France"))
    r = runner(script)
    response = r.invoke(case())
    first = r.evaluate_response(case(), response)
    stricter = case(checks=(CheckSpec(type="exact_match"),))
    second = r.evaluate_response(stricter, response)
    assert script.calls == 1  # one model call, two evaluations
    assert first.status is RecordStatus.PASSED and second.status is RecordStatus.FAILED
    assert first.response == second.response


@pytest.mark.parametrize(
    ("error", "kind", "status"),
    [
        (httpx.ConnectError("refused"), "connection", RequestStatus.FAILED),
        (httpx.ReadTimeout("slow"), "timeout", RequestStatus.TIMEOUT),
    ],
)
def test_model_invocation_failure_is_a_model_error_with_no_substituted_text(error, kind, status) -> None:
    script = Script()
    script.queue.append(error)
    record = runner(script).run_case(case())
    assert record.status is RecordStatus.MODEL_ERROR
    assert record.response.status is status and record.response.text == ""
    assert record.response.error.kind == kind
    assert {m.outcome for m in record.metrics} == {Outcome.NOT_EVALUATED}
    assert "Paris" not in record.response.text  # the reference was not used as a fallback


def test_http_error_status_code_is_recorded() -> None:
    script = Script()
    script.queue.append(httpx.Response(404, json={"error": "model not found"}))
    record = runner(script).run_case(case())
    assert record.status is RecordStatus.MODEL_ERROR
    assert record.response.error.kind == "model_not_found" and record.response.error.status_code == 404


def test_partial_stream_is_not_evaluated_by_default_but_can_be_requested() -> None:
    stream = TARGET.model_copy(update={"stream": True})
    partial = httpx.Response(200, content=(json.dumps({"message": {"role": "assistant", "content": "Paris"}, "done": False})).encode())

    def make(**fields):
        script = Script(tags=("target-model",))
        script.queue.append(partial)
        return EvaluationRunner(client_for(script), config(**fields).model_copy(update={"model": stream}))

    default = make().run_case(case())
    assert default.status is RecordStatus.MODEL_ERROR and default.response.status is RequestStatus.PARTIAL
    assert default.response.text == "Paris"  # kept for diagnosis, not evaluated
    opted_in = make(evaluate_partial=True).run_case(case())
    assert opted_in.status is RecordStatus.PASSED and opted_in.warnings


def test_measurements_are_recorded_and_missing_token_counts_stay_absent() -> None:
    script = Script(tags=("target-model",))
    script.queue.append(httpx.Response(200, json={"message": {"role": "assistant", "content": "Paris"}, "done": True}))
    record = runner(script).run_case(case())
    names = {m.name for m in record.metrics}
    assert {"latency_seconds", "response_chars"} <= names and "output_tokens" not in names


def test_template_renders_prompt_and_context_and_never_leaks_the_reference() -> None:
    script = Script(tags=("target-model",))
    script.queue.append(reply("2014"))
    cfg = config(template=RAG_TEMPLATE)
    rag_case = case(prompt="When?", reference="SECRET-REFERENCE", context=("released in 2014",))
    record = EvaluationRunner(client_for(script), cfg).run_case(rag_case)
    sent = script.payloads()[0]["messages"]
    assert sent[0]["role"] == "system" and "numbered context" in sent[0]["content"]
    assert "[1] released in 2014" in sent[1]["content"] and "When?" in sent[1]["content"]
    assert "SECRET-REFERENCE" not in json.dumps(script.payloads()[0])
    assert record.response.request.template == "rag-basic@1"


def test_without_a_template_the_prompt_is_sent_verbatim_and_context_is_not_injected() -> None:
    script = Script(tags=("target-model",))
    script.queue.append(reply("x"))
    runner(script).run_case(case(context=("hidden context",), system_prompt="sys"))
    sent = script.payloads()[0]["messages"]
    assert sent == [{"role": "system", "content": "sys"}, {"role": "user", "content": "Capital of France?"}]


def test_template_with_unknown_placeholder_is_rejected() -> None:
    bad = PromptTemplate(template_id="t", version="1", user="{prompt} {nope}")
    with pytest.raises(ValueError, match="unknown placeholder"):
        bad.render("q")


def test_generation_settings_are_recorded_in_the_report_config_and_sent_unchanged_for_every_case() -> None:
    generation = GenerationConfig(temperature=0.0, seed=1, max_tokens=64)
    script = Script(tags=("target-model",))
    script.queue.extend([reply("Paris"), reply("Paris")])
    cfg = config().model_copy(update={"model": TARGET.model_copy(update={"generation": generation})})
    report = EvaluationRunner(client_for(script), cfg).run([case(case_id="a"), case(case_id="b")])
    assert [p["options"] for p in script.payloads()] == [{"temperature": 0.0, "num_predict": 64, "seed": 1}] * 2
    assert report.config.model.generation == generation
    assert report.counts["passed"] == 2 and report.passed


def test_run_over_a_dataset_records_dataset_identity() -> None:
    from complete_automation.llm_evaluation import Dataset

    script = Script(tags=("target-model",))
    script.queue.append(reply("Paris"))
    dataset = Dataset(dataset_id="d1", version="v2", fingerprint="f" * 64, cases=(case(),))
    report = runner(script).run(dataset)
    assert (report.dataset_id, report.dataset_fingerprint) == ("d1", "f" * 64)
