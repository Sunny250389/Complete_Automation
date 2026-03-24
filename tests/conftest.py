"""Shared pytest fixtures for DeepEval test automation."""

from __future__ import annotations

from pathlib import Path
import os

import pytest
from dotenv import load_dotenv
from deepeval import login
from deepeval.test_case import LLMTestCase

from llm_automation.config.settings import DATA_DIR, REPORTS_DIR
from llm_automation.core.llm_app import run_llm
from llm_automation.reporting.report_manager import create_run_directory
from llm_automation.utils.excel_reader import read_llm_test_data


@pytest.fixture(scope="session", autouse=True)
def framework_setup() -> Path:
    load_dotenv()
    api_key = os.getenv("CONFIDENT_API_KEY")
    if api_key:
        login(api_key)
    return create_run_directory(REPORTS_DIR)


@pytest.fixture(scope="session")
def llm_test_cases() -> list[LLMTestCase]:
    dataset_path = DATA_DIR / "llm_eval_cases.xlsx"
    rows = read_llm_test_data(dataset_path)
    return [
        LLMTestCase(
            input=row.input,
            expected_output=row.expected_output,
            actual_output=run_llm(row.input),
            retrieval_context=[row.retrieval_context] if row.retrieval_context else None,
        )
        for row in rows
    ]