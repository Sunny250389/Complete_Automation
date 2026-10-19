"""RAG-style cases: required context, grounding versus correctness, context relevance."""

import pytest

from complete_automation.llm_evaluation import (
    CheckSpec,
    CriterionSpec,
    EvaluationRunner,
    JudgeEvaluator,
    LLMDatasetError,
    RecordStatus,
    TestCase,
    parse_cases,
)
from tests.unit.llm_evaluation.helpers import JUDGE, Script, client_for, config, judge_reply, reply

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]

CONTEXT = ("The Zephyr library was first released in 2014 by the Northwind team.",)


def rag_case(**fields) -> TestCase:
    base = dict(case_id="rag1", prompt="When was Zephyr first released?", context=CONTEXT, reference="Zephyr was first released in 2014.")
    return TestCase(**{**base, **fields})


def run(case: TestCase, answer: str, *judge_replies):
    target = Script(tags=("target-model",))
    target.queue.append(reply(answer))
    judge = Script()
    judge.queue.extend(judge_replies)
    runner = EvaluationRunner(client_for(target), config(judge=JUDGE), judge=JudgeEvaluator(client_for(judge), JUDGE))
    return runner.run_case(case), judge


@pytest.mark.parametrize("criterion", ["groundedness", "context_relevance"])
def test_context_criteria_require_context_at_dataset_validation(criterion: str) -> None:
    record = {"case_id": "x", "prompt": "q", "criteria": [{"name": criterion}]}
    with pytest.raises(LLMDatasetError, match=f"{criterion} needs retrieved context"):
        parse_cases([record])


def test_context_overlap_check_requires_context_at_dataset_validation() -> None:
    with pytest.raises(LLMDatasetError, match="needs retrieved context"):
        parse_cases([{"case_id": "x", "prompt": "q", "checks": [{"type": "context_overlap", "minimum": 0.5}]}])


def test_groundedness_judge_sees_the_context_but_not_the_reference() -> None:
    case = rag_case(criteria=(CriterionSpec(name="groundedness", min_score=4),))
    record, judge = run(case, "Zephyr came out in 2014.", judge_reply("groundedness", 5))
    prompt = judge.payloads()[0]["messages"][1]["content"]
    assert "Northwind team" in prompt and "Zephyr came out in 2014." in prompt
    assert "Zephyr was first released in 2014." not in prompt
    assert "<<<REFERENCE" not in prompt and "<<<CONTEXT" in prompt
    assert record.status is RecordStatus.PASSED


def test_correctness_judge_sees_the_reference_but_not_the_context() -> None:
    case = rag_case(criteria=(CriterionSpec(name="correctness", min_score=4),))
    _, judge = run(case, "Zephyr came out in 2014.", judge_reply("correctness", 5))
    prompt = judge.payloads()[0]["messages"][1]["content"]
    assert "<<<REFERENCE" in prompt and "<<<CONTEXT" not in prompt and "Northwind" not in prompt


def test_context_relevance_judges_the_context_without_the_response() -> None:
    case = rag_case(criteria=(CriterionSpec(name="context_relevance", min_score=3),))
    _, judge = run(case, "SHOULD-NOT-BE-SENT", judge_reply("context_relevance", 5))
    prompt = judge.payloads()[0]["messages"][1]["content"]
    assert "<<<CONTEXT" in prompt and "SHOULD-NOT-BE-SENT" not in prompt and "<<<RESPONSE" not in prompt


def test_grounded_but_externally_wrong_answer_is_scored_on_grounding_not_truth() -> None:
    """A response can repeat an error in the source: grounded, yet wrong in the real world."""
    wrong_context = ("The capital of Australia is Sydney.",)
    case = TestCase(case_id="g", prompt="Capital of Australia?", context=wrong_context, reference="Canberra",
                    criteria=(CriterionSpec(name="groundedness", min_score=4), CriterionSpec(name="correctness", min_score=4)))
    record, judge = run(case, "Sydney.", judge_reply("groundedness", 5, rationale="stated in context"),
                        judge_reply("correctness", 1, "fail", "reference says Canberra"))
    outcomes = {m.name: m.outcome.value for m in record.metrics if m.kind == "judge"}
    assert outcomes == {"judge.groundedness": "passed", "judge.correctness": "failed"}
    assert record.status is RecordStatus.FAILED  # failing correctness is not hidden by grounding
    rubric = judge.payloads()[0]["messages"][1]["content"]
    assert "Do not judge real-world correctness" in rubric


def test_reference_based_correctness_without_a_judge_is_deterministic() -> None:
    case = rag_case(checks=(CheckSpec(type="exact_match", normalize_whitespace=True, case_sensitive=False),))
    target = Script(tags=("target-model",))
    target.queue.append(reply("  zephyr was first   released in 2014.\n"))
    record = EvaluationRunner(client_for(target), config()).run_case(case)
    assert record.status is RecordStatus.PASSED and not record.judge_results


def test_lexical_context_overlap_flags_an_answer_unrelated_to_the_context() -> None:
    case = rag_case(checks=(CheckSpec(type="context_overlap", minimum=0.4),))
    target = Script(tags=("target-model",))
    target.queue.extend([reply("Zephyr was released in 2014 by Northwind."), reply("Bananas ripen quickly in warm weather.")])
    runner = EvaluationRunner(client_for(target), config())
    assert runner.run_case(case).status is RecordStatus.PASSED
    assert runner.run_case(case).status is RecordStatus.FAILED
