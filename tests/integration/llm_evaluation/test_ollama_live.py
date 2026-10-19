"""Real-model tests: these call an actual, locally installed Ollama model.

They are opt-in. Enable with ``--cauto-env ollama_local`` (or ``CAUTO_MODULES__LLM_EVALUATION__LIVE=true``)
and name the model: ``CAUTO_MODULES__LLM_EVALUATION__MODEL=<name from `ollama list`>``. Without that
they skip with the reason (they fail instead if ``require_model`` is true). A response is only ever
evaluated if the model produced it; nothing here substitutes a fixture. See docs/llm_evaluation.md.
"""

import pytest

from complete_automation.llm_evaluation import (
    CheckSpec,
    CriterionSpec,
    EvaluationConfig,
    GenerationConfig,
    RecordStatus,
    TestCase,
    ThresholdRule,
)
from complete_automation.llm_evaluation.schemas import RequestStatus

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.llm_live, pytest.mark.cauto_module("llm_evaluation")]

# think is left unset: reasoning models then return their reasoning in a separate field.
GENERATION = GenerationConfig(temperature=0.0, seed=1, max_tokens=400)
# Reasoning models can spend the whole budget thinking and return an empty answer (finish reason
# "length"); the dataset test gives them more room, and the `completed` check reports it if not enough.
DATASET_GENERATION = GenerationConfig(temperature=0.0, seed=1, max_tokens=1500)
# The judge replies with a short JSON object, so its reasoning is switched off to keep it quick.
JUDGE_GENERATION = GenerationConfig(temperature=0.0, seed=1, think=False)


def test_real_ollama_model_response_is_captured_and_evaluated(llm_settings, llm_live_model, llm_evaluate, result_recorder) -> None:
    case = TestCase(
        case_id="capital-france",
        prompt="What is the capital of France? Answer in one short sentence.",
        checks=(
            CheckSpec(type="not_empty"),
            CheckSpec(type="completed"),
            CheckSpec(type="contains", value="paris", case_sensitive=False),
        ),
        thresholds=(ThresholdRule(metric="latency_seconds", operator="le", value=llm_settings.timeout_seconds),),
    )
    config = EvaluationConfig(
        model=llm_settings.target_model(GENERATION),
        suite_thresholds=(ThresholdRule(metric="case_pass_rate", operator="ge", value=1.0),),
    )

    report = llm_evaluate([case], config, dataset_id="live-smoke", assert_passed=True)

    (record,) = report.records
    response = record.response
    # The evidence that a real model produced this: provider/model from the server round trip,
    # server-measured durations, and a digest from /api/tags, none of which a fixture would carry.
    assert response.status is RequestStatus.COMPLETED and response.text.strip()
    assert response.model == llm_settings.model and response.provider == "ollama"
    assert response.model_identifier == llm_live_model.identifier
    assert response.output_tokens and response.provider_metadata.get("total_duration_ns")
    assert "paris" in response.text.lower()
    assert record.status is RecordStatus.PASSED and report.passed
    print(
        f"\nOllama model actually used: {response.model} (digest {response.model_identifier}) at "
        f"{config.model.base_url}; {response.output_tokens} output tokens in {response.latency_seconds:.1f}s; "
        f"response: {response.text.strip()[:120]!r}"
    )
    assert report.artifacts


def test_real_model_dataset_with_structured_output(llm_settings, llm_live_model, llm_load_dataset, llm_evaluate) -> None:
    """Runs the shipped smoke dataset. Failures here are real findings about the model, not the framework."""
    dataset = llm_load_dataset("examples/ollama_smoke.jsonl")
    config = EvaluationConfig(model=llm_settings.target_model(DATASET_GENERATION))
    report = llm_evaluate(dataset, config)
    assert len(report.records) == len(dataset.cases)
    assert all(r.response.status is RequestStatus.COMPLETED for r in report.records), report.failure_reasons
    report.assert_passed()


def test_real_judge_scores_a_real_model_answer(llm_settings, llm_live_model, llm_live_judge, llm_evaluate) -> None:
    case = TestCase(
        case_id="judged-capital", prompt="What is the capital of France? Answer in one short sentence.",
        reference="Paris is the capital of France.",
        criteria=(CriterionSpec(name="correctness", min_score=3), CriterionSpec(name="relevance")),
    )
    config = EvaluationConfig(model=llm_settings.target_model(GENERATION), judge=llm_settings.judge(JUDGE_GENERATION))
    report = llm_evaluate([case], config)
    (record,) = report.records
    assert record.response.model == llm_settings.model and record.response.text.strip()
    assert [j.judge_model for j in record.judge_results] == [llm_settings.judge_model] * 2
    # The judge is an estimate; this asserts it produced valid structured output, then applies the gate.
    assert all(j.usable for j in record.judge_results), [j.error for j in record.judge_results]
    report.assert_passed()
