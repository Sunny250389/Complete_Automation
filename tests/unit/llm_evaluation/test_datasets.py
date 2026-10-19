"""Dataset loading and validation: formats, identity, and what each evaluation requires."""

import json
from pathlib import Path

import pytest

from complete_automation.llm_evaluation import LLMDatasetError, load_dataset, parse_cases

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]

ROOT = Path(__file__).resolve().parents[3] / "data" / "llm_evaluation"


def write(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def record(**fields) -> dict:
    return {"case_id": "c1", "prompt": "q", "checks": [{"type": "not_empty"}], **fields}


def test_jsonl_dataset_loads_with_identity_and_fingerprint(tmp_path: Path) -> None:
    write(tmp_path, "cases.jsonl", "\n".join(json.dumps(record(case_id=f"c{i}")) for i in range(3)) + "\n\n")
    dataset = load_dataset(tmp_path, "cases.jsonl", version="v1")
    assert (dataset.dataset_id, dataset.version, len(dataset.cases)) == ("cases", "v1", 3)
    assert len(dataset.fingerprint) == 64
    write(tmp_path, "other.jsonl", json.dumps(record()) + "\n")
    assert load_dataset(tmp_path, "other.jsonl").fingerprint != dataset.fingerprint


def test_json_yaml_and_csv_are_supported_through_the_shared_loader(tmp_path: Path) -> None:
    write(tmp_path, "a.json", json.dumps([record()]))
    write(tmp_path, "b.yaml", "- case_id: y1\n  prompt: hello\n  checks:\n    - type: not_empty\n")
    write(tmp_path, "c.csv", 'case_id,prompt,reference,checks,tags\nz1,hi,there,"[{""type"": ""exact_match""}]","[""t""]"\n')
    assert load_dataset(tmp_path, "a.json").cases[0].case_id == "c1"
    assert load_dataset(tmp_path, "b.yaml").cases[0].prompt == "hello"
    csv_case = load_dataset(tmp_path, "c.csv").cases[0]
    assert csv_case.reference == "there" and csv_case.checks[0].type == "exact_match" and csv_case.tags == ("t",)


def test_malformed_jsonl_names_the_line_not_the_content(tmp_path: Path) -> None:
    write(tmp_path, "bad.jsonl", json.dumps(record()) + "\n{broken secret-value\n")
    with pytest.raises(LLMDatasetError, match="line 2") as info:
        load_dataset(tmp_path, "bad.jsonl")
    assert "secret-value" not in str(info.value)


def test_path_outside_the_data_root_and_missing_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(LLMDatasetError, match="outside"):
        load_dataset(tmp_path, "../escape.jsonl")
    with pytest.raises(LLMDatasetError, match="not found"):
        load_dataset(tmp_path, "nope.jsonl")


@pytest.mark.parametrize(
    ("records", "fragment"),
    [
        ([{"case_id": "x", "checks": [{"type": "not_empty"}]}], "prompt"),
        ([record(prompt="")], "prompt"),
        ([record(), record()], "duplicate case_id"),
        ([record(case_id="bad id!")], "case_id"),
        ([record(checks=[{"type": "nonsense"}])], "checks"),
        ([record(checks=[{"type": "regex", "value": "("}])], "invalid regular expression"),
        ([record(checks=[{"type": "json_schema", "value": {"type": "bogus"}}])], "invalid JSON Schema"),
        ([record(checks=[{"type": "json_schema"}])], "needs a schema"),
        ([record(checks=[{"type": "exact_match"}])], "needs a 'value' or a case reference"),
        ([record(checks=[{"type": "contains", "value": "a"}, {"type": "contains", "value": "b"}])], "check names must be unique"),
        ([record(criteria=[{"name": "correctness"}])], "needs a reference answer"),
        ([record(criteria=[{"name": "groundedness"}])], "needs retrieved context"),
        ([record(criteria=[{"name": "vibes"}])], "unknown criterion"),
        ([record(criteria=[{"name": "clarity"}, {"name": "clarity"}])], "duplicate criteria"),
        ([record(checks=[], criteria=[])], "nothing to evaluate"),
        ([record(thresholds=[{"metric": "m", "operator": "approx", "value": 1}])], "thresholds"),
        ([record(unexpected_field=1)], "unexpected_field"),
        (["not an object"], "not an object"),
    ],
)
def test_invalid_records_are_rejected_before_execution(records: list, fragment: str) -> None:
    with pytest.raises(LLMDatasetError, match=fragment):
        parse_cases(records)


def test_all_problems_are_listed_together() -> None:
    with pytest.raises(LLMDatasetError) as info:
        parse_cases([record(case_id="a", criteria=[{"name": "correctness"}]), {"case_id": "b"}])
    message = str(info.value)
    assert "record 2" in message and message.count("\n") >= 2


def test_non_list_data_is_rejected() -> None:
    with pytest.raises(LLMDatasetError, match="list of records"):
        parse_cases({"case_id": "x"})


def test_context_given_as_a_single_string_becomes_a_one_item_context() -> None:
    (case,) = parse_cases([record(context="just one passage")])
    assert case.context == ("just one passage",)


@pytest.mark.parametrize("name", ["ollama_smoke.jsonl", "rag_cases.jsonl"])
def test_shipped_example_datasets_are_valid(name: str) -> None:
    dataset = load_dataset(ROOT, f"examples/{name}")
    assert dataset.cases and len({c.case_id for c in dataset.cases}) == len(dataset.cases)
