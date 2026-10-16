"""Read xlsx datasets without requiring pandas/openpyxl dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


@dataclass
class LLMTestData:
    input: str
    expected_output: str
    retrieval_context: str


def read_llm_test_data(path: Path) -> list[LLMTestData]:
    """Load rows from a minimal xlsx file with fixed column order."""
    with zipfile.ZipFile(path) as workbook:
        shared_strings = _read_shared_strings(workbook)
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = sheet.findall(".//x:sheetData/x:row", namespace)
    parsed: list[LLMTestData] = []

    for row in rows[1:]:  # skip header
        values: list[str] = []
        for cell in row.findall("x:c", namespace):
            cell_type = cell.attrib.get("t")
            value_node = cell.find("x:v", namespace)
            raw = "" if value_node is None else value_node.text or ""
            if cell_type == "s" and raw:
                values.append(shared_strings[int(raw)])
            else:
                values.append(raw)

        if len(values) >= 3:
            parsed.append(
                LLMTestData(
                    input=values[0],
                    expected_output=values[1],
                    retrieval_context=values[2],
                )
            )

    return parsed


def _read_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    return [node.find("x:t", namespace).text or "" for node in root.findall("x:si", namespace)]