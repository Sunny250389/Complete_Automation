"""LLM evaluation rows from an xlsx workbook.

Kept for the LLM suite. Workbook parsing now lives in
``complete_automation.core.datasets``; this module only maps rows to
``LLMTestData``. Columns are read by position (input, expected output,
retrieval context); extra columns such as ``case_id`` are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from complete_automation.core.datasets import read_xlsx_rows


@dataclass
class LLMTestData:
    input: str
    expected_output: str
    retrieval_context: str


def read_llm_test_data(path: Path) -> list[LLMTestData]:
    """Load rows from an xlsx file with fixed column order, skipping the header row."""
    parsed: list[LLMTestData] = []
    for values in read_xlsx_rows(path)[1:]:
        if len(values) >= 3:
            parsed.append(
                LLMTestData(
                    input=values[0],
                    expected_output=values[1],
                    retrieval_context=values[2],
                )
            )
    return parsed
