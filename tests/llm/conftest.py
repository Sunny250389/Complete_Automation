"""Shared pytest fixtures for DeepEval test automation.

The suite needs a judge model: DeepEval's default is an OpenAI model, so it runs only when
``OPENAI_API_KEY`` is set (in the environment or ``.env``) and is skipped, with that reason,
otherwise. Its tests are marked ``live`` like the framework's other external-service tests.
"""

from __future__ import annotations

from pathlib import Path
import os

import pytest
from dotenv import dotenv_values, find_dotenv, load_dotenv
from deepeval import login
from deepeval.test_case import LLMTestCase

from complete_automation.config.settings import DATA_DIR, REPORTS_DIR
from complete_automation.llm.app import run_llm
from complete_automation.reporting.report_manager import create_run_directory
from complete_automation.utils.excel_reader import read_llm_test_data


@pytest.fixture(scope="session", autouse=True)
def framework_setup() -> Path:
    load_dotenv()
    api_key = os.getenv("CONFIDENT_API_KEY")
    if api_key:
        login(api_key)
    return create_run_directory(REPORTS_DIR)


def _judge_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or dotenv_values(find_dotenv(usecwd=True)).get("OPENAI_API_KEY"))


@pytest.fixture(scope="session")
def llm_test_cases() -> list[LLMTestCase]:
    if not _judge_configured():
        pytest.skip("DeepEval judge not configured: set OPENAI_API_KEY (DeepEval's default judge is an OpenAI model)")
    dataset_path = DATA_DIR / "eval_cases.xlsx"
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