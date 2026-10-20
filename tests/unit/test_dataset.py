"""Offline checks on the seed LLM dataset and its xlsx reader. No network, no API keys."""

import json

import pytest

from complete_automation.config.settings import DATA_DIR
from complete_automation.utils.excel_reader import read_llm_test_data

pytestmark = pytest.mark.cauto_module("llm")


def test_data_dir_is_inside_the_repository():
    assert DATA_DIR.is_dir()
    assert (DATA_DIR / "eval_cases.xlsx").is_file()


def test_xlsx_rows_match_json_twin():
    rows = read_llm_test_data(DATA_DIR / "eval_cases.xlsx")
    twin = json.loads((DATA_DIR / "eval_cases.json").read_text(encoding="utf-8"))

    assert [(r.input, r.expected_output, r.retrieval_context) for r in rows] == [
        (case["input"], case["expected_output"], case["retrieval_context"]) for case in twin
    ]


def test_every_case_has_all_fields_filled():
    rows = read_llm_test_data(DATA_DIR / "eval_cases.xlsx")

    assert len(rows) == 6
    for row in rows:
        assert row.input and row.expected_output and row.retrieval_context
