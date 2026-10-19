"""Deterministic checks with known expected outcomes. No model, no network."""

import pytest

from complete_automation.llm_evaluation import CheckSpec, ModelConfig, ModelRequest, ModelResponse, TestCase
from complete_automation.llm_evaluation.evaluators.deterministic import run_check
from complete_automation.llm_evaluation.schemas import Message, Outcome, RequestStatus

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]

REQUEST = ModelRequest(messages=(Message(role="user", content="q"),), config=ModelConfig(model="m"))


def response(text: str, **fields) -> ModelResponse:
    return ModelResponse(provider="ollama", model="m", request=REQUEST, status=fields.pop("status", RequestStatus.COMPLETED),
                         text=text, latency_seconds=0.1, **fields)


def case(**fields) -> TestCase:
    return TestCase(case_id="c1", prompt="q", **fields)


def outcome(spec: dict, text: str, case_: TestCase | None = None, **fields) -> Outcome:
    return run_check(CheckSpec(**spec), case_ or case(), response(text, **fields)).outcome


P, F, E = Outcome.PASSED, Outcome.FAILED, Outcome.ERROR


@pytest.mark.parametrize(
    ("spec", "text", "expected"),
    [
        ({"type": "exact_match", "value": "Paris"}, "Paris", P),
        ({"type": "exact_match", "value": "Paris"}, "paris", F),
        ({"type": "exact_match", "value": "Paris"}, " Paris\n", F),  # no hidden normalisation
        ({"type": "exact_match", "value": "Paris", "normalize_whitespace": True}, " Paris\n", P),
        ({"type": "exact_match", "value": "a b", "normalize_whitespace": True}, "a   b", P),
        ({"type": "exact_match", "value": "Paris", "case_sensitive": False}, "PARIS", P),
        ({"type": "exact_match", "value": "Paris"}, "Paris.", F),  # punctuation is never stripped
        ({"type": "not_empty"}, "x", P),
        ({"type": "not_empty"}, "  \n", F),
        ({"type": "contains", "value": "paris", "case_sensitive": False}, "It is Paris.", P),
        ({"type": "contains", "value": "paris"}, "It is Paris.", F),
        ({"type": "contains", "value": ["Summary", "Risks"]}, "Summary ... Risks", P),
        ({"type": "contains", "value": ["Summary", "Risks"]}, "Summary only", F),
        ({"type": "not_contains", "value": ["password", "secret"]}, "all fine", P),
        ({"type": "not_contains", "value": ["password", "secret"]}, "the secret is", F),
        ({"type": "regex", "value": r"^\d{3}-\d{4}$"}, "555-1234", P),
        ({"type": "regex", "value": r"^\d{3}-\d{4}$"}, "5551234", F),
        ({"type": "regex", "value": "hello", "case_sensitive": False}, "HELLO", P),
        ({"type": "json_valid"}, '{"a": 1}', P),
        ({"type": "json_valid"}, '```json\n{"a": 1}\n```', P),
        ({"type": "json_valid"}, '{"a": 1,}', F),
        ({"type": "json_valid"}, "", F),
        ({"type": "length", "maximum": 5}, "abcde", P),
        ({"type": "length", "maximum": 5}, "abcdef", F),
        ({"type": "length", "minimum": 3}, "ab", F),
        ({"type": "length", "minimum": 2, "maximum": 3, "unit": "words"}, "one two", P),
        ({"type": "length", "maximum": 2, "unit": "words"}, "one two three", F),
        ({"type": "refusal", "value": True}, "I'm sorry, but I can't help with that.", P),
        ({"type": "refusal", "value": True}, "Sure, here you go.", F),
        ({"type": "refusal", "value": False}, "Sure, here you go.", P),
        ({"type": "refusal", "value": False}, "I cannot help with that request.", F),
        ({"type": "refusal", "value": True, "patterns": ["nope"]}, "nope.", P),
    ],
)
def test_text_checks(spec: dict, text: str, expected: Outcome) -> None:
    assert outcome(spec, text) is expected


def test_exact_match_uses_case_reference_when_no_value_given() -> None:
    assert outcome({"type": "exact_match"}, "Paris", case(reference="Paris")) is P
    assert outcome({"type": "exact_match"}, "Rome", case(reference="Paris")) is F


@pytest.mark.parametrize(
    ("spec", "case_"),
    [
        ({"type": "exact_match"}, case()),  # no value and no reference
        ({"type": "contains"}, case()),
        ({"type": "contains", "value": []}, case()),
        ({"type": "contains", "value": [1]}, case()),
        ({"type": "regex", "value": "("}, case()),
        ({"type": "regex"}, case()),
        ({"type": "length"}, case()),
        ({"type": "refusal"}, case()),
        ({"type": "json_schema"}, case()),
        ({"type": "json_schema", "value": {"type": "nonsense"}}, case()),
        ({"type": "json_keys"}, case()),
        ({"type": "context_overlap", "minimum": 0.5}, case()),  # no context
        ({"type": "context_overlap"}, case(context=("text",))),  # no explicit minimum
    ],
)
def test_checks_that_cannot_be_evaluated_are_errors_never_passes(spec: dict, case_: TestCase) -> None:
    result = run_check(CheckSpec(**spec), case_, response("anything"))
    assert result.outcome is E and result.explanation


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                   "tags": {"type": "array", "items": {"type": "string"}},
                   "address": {"type": "object", "properties": {"zip": {"type": "string"}}, "required": ["zip"]}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("text", "expected", "fragment"),
    [
        ('{"name": "A", "age": 3}', P, "conforms"),
        ('{"name": "A", "age": 3, "address": {"zip": "1"}}', P, "conforms"),
        ('{"name": "A"}', F, "age"),  # missing key
        ('{"name": "A", "age": "3"}', F, "age"),  # wrong type
        ('{"name": "A", "age": 3, "extra": 1}', F, "extra"),  # unexpected key
        ('{"name": "A", "age": 3, "address": {}}', F, "address"),  # nested
        ('{"name": "A", "age": 3, "tags": [1]}', F, "tags"),
        ("not json", F, "not valid JSON"),  # distinguished from a schema violation
    ],
)
def test_json_schema_distinguishes_invalid_json_from_schema_violations(text: str, expected: Outcome, fragment: str) -> None:
    result = run_check(CheckSpec(type="json_schema", value=SCHEMA), case(), response(text))
    assert result.outcome is expected and fragment in result.explanation
    assert bool(result.details.get("invalid_json")) is (text == "not json")


def test_json_schema_defaults_to_case_expected_schema() -> None:
    case_ = case(expected_schema=SCHEMA)
    assert run_check(CheckSpec(type="json_schema"), case_, response('{"name": "A", "age": 1}')).outcome is P


def test_json_keys_reports_missing_and_unexpected_keys() -> None:
    spec = CheckSpec(type="json_keys", value=["a", "b"], allowed_keys=("c",))
    assert run_check(spec, case(), response('{"a": 1, "b": 2, "c": 3}')).outcome is P
    result = run_check(spec, case(), response('{"a": 1, "z": 2}'))
    assert result.outcome is F and result.details == {"missing": ["b"], "unexpected": ["z"]}
    assert run_check(spec, case(), response("[1]")).outcome is F
    assert run_check(CheckSpec(type="json_keys", value=["a"]), case(), response('{"a": 1, "z": 2}')).outcome is P


def test_completed_check_detects_truncation_and_partial_responses() -> None:
    spec = {"type": "completed"}
    assert outcome(spec, "x", finish_reason="stop") is P
    assert outcome(spec, "x", finish_reason="length") is F
    assert outcome(spec, "x", status=RequestStatus.PARTIAL) is F
    result = run_check(CheckSpec(**spec), case(), response("x"))  # no finish reason exposed
    assert result.outcome is P and "cannot be ruled out" in result.explanation


def test_context_overlap_is_a_lexical_ratio_with_an_explicit_minimum() -> None:
    case_ = case(context=("Zephyr was released in 2014 by Northwind",))
    spec = CheckSpec(type="context_overlap", minimum=0.5)
    good = run_check(spec, case_, response("Zephyr released 2014 Northwind"))
    bad = run_check(spec, case_, response("Quantum cats compute banana smoothies"))
    assert good.outcome is P and good.value == 1.0
    assert bad.outcome is F and bad.value == 0.0 and "lexical proxy" in bad.explanation


def test_check_name_overrides_the_type_as_metric_name() -> None:
    result = run_check(CheckSpec(type="contains", value="x", name="has_x"), case(), response("x"))
    assert result.name == "has_x"
