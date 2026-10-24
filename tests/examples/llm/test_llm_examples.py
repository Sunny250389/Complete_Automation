"""Examples of the LLM evaluation module, written the way a team would write model tests.

Test cases live in ``data/llm_evaluation/examples/`` and are read with the ``llm_load_dataset`` fixture,
which validates every case before anything runs. The offline example needs no model. The live example
calls the model named in ``modules.llm_evaluation.model`` and skips, with the reason, unless that is set
(see docs/llm_evaluation.md). A response is only ever evaluated if the model produced it.
"""

import pytest

from complete_automation.llm_evaluation import EvaluationConfig, GenerationConfig

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]

SMOKE = "examples/ollama_smoke.jsonl"
GENERATION = GenerationConfig(temperature=0.0, seed=1, max_tokens=800)


def test_the_example_dataset_loads_and_validates(llm_load_dataset) -> None:
    dataset = llm_load_dataset(SMOKE)

    assert [case.case_id for case in dataset.cases] == ["capital-france", "json-city", "arithmetic"]
    assert all(case.checks for case in dataset.cases)  # every case says what it evaluates


@pytest.mark.llm_live
def test_a_real_model_answers_the_example_dataset(llm_settings, llm_live_model, llm_load_dataset, llm_evaluate) -> None:
    dataset = llm_load_dataset(SMOKE)
    config = EvaluationConfig(model=llm_settings.target_model(GENERATION))

    report = llm_evaluate(dataset, config, dataset_id="ollama-smoke")

    assert len(report.records) == len(dataset.cases)
    for record in report.records:
        # The answer is the model's own output: captured once, with the model's name and digest.
        assert record.response.text.strip(), record.case_id
        assert record.response.model == llm_settings.model
        assert record.response.model_identifier == llm_live_model.identifier
    # Gates and checks are reported per record. Pass assert_passed=True to make a failed check fail the test.
