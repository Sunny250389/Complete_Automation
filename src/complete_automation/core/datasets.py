"""Generic test data loading, validation and deterministic factories.

Supported formats: JSON, YAML (safe loader only), CSV and xlsx (standard library
only; first worksheet). :class:`DataLoader` confines loading to one data root so
tests cannot read arbitrary files by accident. Validation uses pydantic models;
errors report the record index and field, never the offending value.
"""

from __future__ import annotations

import csv
import json
import random
import re
import string
import uuid
import zipfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from complete_automation.core.exceptions import TestDataError
from complete_automation.core.paths import UnsafePathError, resolve_within

ModelT = TypeVar("ModelT", bound=BaseModel)
T = TypeVar("T")

_XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise TestDataError(f"test data file not found: {path}")
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise TestDataError(f"cannot read test data file {path}") from error


def load_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(Path(path)))
    except json.JSONDecodeError as error:
        raise TestDataError(
            f"malformed JSON in {path} at line {error.lineno}, column {error.colno}"
        ) from error


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(_read_text(Path(path)))
    except yaml.YAMLError as error:
        raise TestDataError(f"malformed YAML in {path}") from error


def load_csv(path: Path) -> list[dict[str, str]]:
    """Rows as dicts keyed by the header row; ragged rows are rejected."""
    text = _read_text(Path(path))
    reader = csv.DictReader(text.splitlines(), restkey="__extra__", restval=None)
    if not reader.fieldnames:
        raise TestDataError(f"CSV file has no header row: {path}")
    headers = [h.strip() for h in reader.fieldnames]
    if len(set(headers)) != len(headers) or "" in headers:
        raise TestDataError(f"CSV header has blank or duplicate column names: {path}")
    rows: list[dict[str, str]] = []
    try:
        for line_number, row in enumerate(reader, start=2):
            if "__extra__" in row or any(value is None for value in row.values()):
                raise TestDataError(
                    f"CSV row on line {line_number} of {path} has "
                    f"a different number of fields than the header"
                )
            rows.append({key.strip(): value for key, value in row.items()})
    except csv.Error as error:
        raise TestDataError(f"malformed CSV in {path}") from error
    return rows


def read_xlsx_rows(path: Path) -> list[list[str]]:
    """Every row of the first worksheet as strings, with blank cells kept in place."""
    path = Path(path)
    if not path.is_file():
        raise TestDataError(f"test data file not found: {path}")
    try:
        with zipfile.ZipFile(path) as workbook:
            names = set(workbook.namelist())
            shared = (
                _shared_strings(workbook.read("xl/sharedStrings.xml"))
                if "xl/sharedStrings.xml" in names
                else []
            )
            sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError) as error:
        raise TestDataError(f"not a readable xlsx workbook: {path}") from error

    rows: list[list[str]] = []
    for row in sheet.findall(".//x:sheetData/x:row", _XLSX_NS):
        values: dict[int, str] = {}
        next_column = 0
        for cell in row.findall("x:c", _XLSX_NS):
            column = _column_index(cell.attrib.get("r")) if cell.attrib.get("r") else next_column
            values[column] = _cell_text(cell, shared, path)
            next_column = column + 1
        width = max(values) + 1 if values else 0
        rows.append([values.get(i, "") for i in range(width)])
    return rows


def load_xlsx(path: Path) -> list[dict[str, str]]:
    """Rows of the first worksheet as dicts keyed by the header row."""
    rows = read_xlsx_rows(path)
    if not rows:
        return []
    headers = [h.strip() for h in rows[0]]
    if "" in headers or len(set(headers)) != len(headers):
        raise TestDataError(f"xlsx header has blank or duplicate column names: {path}")
    records = []
    for index, row in enumerate(rows[1:], start=2):
        if len(row) > len(headers) and any(row[len(headers):]):
            raise TestDataError(f"xlsx row {index} of {path} has values beyond the header")
        if not any(row):
            continue
        padded = row + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded)))
    return records


def _column_index(reference: str | None) -> int:
    match = _CELL_REF.match(reference or "")
    if not match:
        raise TestDataError(f"invalid cell reference {reference!r}")
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _cell_text(cell: ET.Element, shared: list[str], path: Path) -> str:
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iterfind(".//x:t", _XLSX_NS))
    value = cell.find("x:v", _XLSX_NS)
    raw = "" if value is None or value.text is None else value.text
    if kind == "s" and raw:
        try:
            return shared[int(raw)]
        except (ValueError, IndexError) as error:
            raise TestDataError(f"broken shared-string reference in {path}") from error
    return raw


def _shared_strings(xml: bytes) -> list[str]:
    root = ET.fromstring(xml)
    return [
        "".join(t.text or "" for t in item.iterfind(".//x:t", _XLSX_NS))
        for item in root.findall("x:si", _XLSX_NS)
    ]


_LOADERS = {
    ".json": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
}


class DataLoader:
    """Loads test data files that live under one root directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def path(self, relative: str | Path) -> Path:
        try:
            return resolve_within(self.root, relative)
        except UnsafePathError as error:
            raise TestDataError(f"test data path outside {self.root}: {error}") from error

    def load(self, relative: str | Path) -> Any:
        """Load a file by its extension (.json, .yaml/.yml, .csv, .xlsx)."""
        path = self.path(relative)
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            raise TestDataError(f"unsupported test data format {path.suffix!r}: {path}")
        return loader(path)

    def load_records(self, relative: str | Path, model: type[ModelT]) -> list[ModelT]:
        return validate_records(self.load(relative), model, source=str(relative))


def validate_records(data: Any, model: type[ModelT], *, source: str = "data") -> list[ModelT]:
    """Validate a list of records against ``model``; errors name record and field only."""
    if not isinstance(data, list):
        raise TestDataError(f"{source}: expected a list of records, got {type(data).__name__}")
    try:
        return TypeAdapter(list[model]).validate_python(data)  # type: ignore[valid-type]
    except ValidationError as error:
        problems = []
        for item in error.errors(include_input=False, include_url=False):
            location = item["loc"]
            record = f"record {location[0]}" if location else "data"
            field = ".".join(str(p) for p in location[1:]) or "<record>"
            problems.append(f"  - {record}, {field}: {item['msg']}")
        raise TestDataError(f"{source} failed validation:\n" + "\n".join(problems)) from None


class DataFactory:
    """Seeded, reproducible generator of simple test values.

    Two factories with the same seed produce the same sequence of values.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self._counters: dict[str, int] = {}

    def integer(self, low: int = 0, high: int = 1_000_000) -> int:
        return self.random.randint(low, high)

    def string(self, length: int = 12, alphabet: str = string.ascii_lowercase + string.digits) -> str:
        if length < 0:
            raise ValueError("length must not be negative")
        return "".join(self.random.choice(alphabet) for _ in range(length))

    def choice(self, options: Sequence[T]) -> T:
        if not options:
            raise ValueError("options must not be empty")
        return self.random.choice(options)

    def email(self, domain: str = "example.test") -> str:
        return f"{self.string(10)}@{domain}"

    def uuid(self) -> uuid.UUID:
        return uuid.UUID(int=self.random.getrandbits(128), version=4)

    def sequence(self, prefix: str, width: int = 4) -> str:
        """``prefix-0001``, ``prefix-0002``, ... counted per prefix."""
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]:0{width}d}"
