"""Loading and validating LLM evaluation datasets.

Formats: ``.jsonl`` (one test case per line, added here because the shared loader has none) plus
``.json`` / ``.yaml`` / ``.yml`` / ``.csv`` / ``.xlsx`` through the shared :class:`DataLoader`.
Loading is confined to one data root. Errors name the record and field, never the content.

:func:`validate_cases` checks what a case needs for the evaluation it asks for: a reference for
reference-based checks and criteria, context for context-based ones, valid regular expressions and
JSON Schemas, unique case ids and unique check names within a case.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from complete_automation.core.datasets import DataLoader
from complete_automation.core.exceptions import TestDataError
from complete_automation.core.paths import UnsafePathError, resolve_within
from complete_automation.llm_evaluation.errors import LLMDatasetError
from complete_automation.llm_evaluation.evaluators.rubrics import resolve_rubric
from complete_automation.llm_evaluation.schemas import Dataset, TestCase

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]

_JSON_LIST_FIELDS = ("checks", "criteria", "thresholds", "tags", "context", "expected_schema", "metadata")
_MAX_BYTES = 20_000_000


def load_dataset(root: Path, relative: str | Path, *, version: str | None = None) -> Dataset:
    """Load, validate and fingerprint a dataset file located under ``root``."""
    try:
        path = resolve_within(Path(root).resolve(), relative)
    except UnsafePathError as error:
        raise LLMDatasetError(f"dataset path outside {root}: {error}") from error
    if not path.is_file():
        raise LLMDatasetError(f"dataset file not found: {path}")
    if path.stat().st_size > _MAX_BYTES:
        raise LLMDatasetError(f"dataset file is larger than {_MAX_BYTES} bytes: {path}")
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == ".jsonl":
            records = _read_jsonl(raw.decode("utf-8-sig"), str(relative))
        else:
            records = DataLoader(Path(root)).load(relative)
    except (OSError, UnicodeDecodeError) as error:
        raise LLMDatasetError(f"cannot read dataset {relative}") from error
    except TestDataError as error:
        raise LLMDatasetError(str(error)) from error
    cases = parse_cases(records, source=str(relative))
    return Dataset(
        dataset_id=path.stem,
        version=version or "unversioned",
        fingerprint=hashlib.sha256(raw).hexdigest(),
        cases=tuple(cases),
    )


def _read_jsonl(text: str, source: str) -> list[Any]:
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise LLMDatasetError(f"{source}: line {number} is not valid JSON ({error.msg})") from None
    return records


def parse_cases(records: Any, *, source: str = "data") -> list[TestCase]:
    """Turn raw records into validated cases; every problem is listed, none is silently dropped."""
    if not isinstance(records, list):
        raise LLMDatasetError(f"{source}: expected a list of records, got {type(records).__name__}")
    cases: list[TestCase] = []
    problems: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            problems.append(f"record {index}: not an object")
            continue
        try:
            cases.append(TestCase.model_validate(_decode_cells(record)))
        except ValidationError as error:
            for item in error.errors(include_input=False, include_url=False):
                field = ".".join(str(p) for p in item["loc"]) or "<record>"
                problems.append(f"record {index} ({record.get('case_id', '?')}), {field}: {item['msg']}")
    problems += _case_problems(cases)
    if problems:
        raise LLMDatasetError(f"{source} failed validation:\n  " + "\n  ".join(problems))
    return cases


def _decode_cells(record: dict[str, Any]) -> dict[str, Any]:
    """CSV/xlsx cells hold structured fields as JSON text; decode them. Empty cells mean 'absent'."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, str) and not value.strip() and key != "prompt":
            continue
        if isinstance(value, str) and key in _JSON_LIST_FIELDS and value.lstrip()[:1] in ("[", "{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # left as text; model validation then reports the field
        out[key] = value
    return out


_REFERENCE_CHECKS = {"exact_match"}


def validate_cases(cases: Sequence[TestCase], *, source: str = "data") -> None:
    problems = _case_problems(cases)
    if problems:
        raise LLMDatasetError(f"{source} failed validation:\n  " + "\n  ".join(problems))


def _case_problems(cases: Sequence[TestCase]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        where = f"case {case.case_id}"
        if case.case_id in seen:
            problems.append(f"{where}: duplicate case_id")
        seen.add(case.case_id)
        labels = [c.label for c in case.checks]
        if len(labels) != len(set(labels)):
            problems.append(f"{where}: check names must be unique (set 'name' on repeated check types)")
        for check in case.checks:
            if check.type in _REFERENCE_CHECKS and check.value is None and case.reference is None:
                problems.append(f"{where}: {check.label} needs a 'value' or a case reference")
            if check.type == "json_schema" and check.value is None and case.expected_schema is None:
                problems.append(f"{where}: {check.label} needs a schema ('value' or expected_schema)")
            if check.type == "context_overlap" and not case.context:
                problems.append(f"{where}: {check.label} needs retrieved context")
            if check.type == "regex":
                try:
                    re.compile(str(check.value))
                except re.error as error:
                    problems.append(f"{where}: {check.label} has an invalid regular expression ({error})")
        schemas = [c.value for c in case.checks if c.type == "json_schema" and c.value is not None]
        schemas += [case.expected_schema] if case.expected_schema is not None else []
        for schema in schemas:
            if jsonschema is not None and isinstance(schema, dict):
                try:
                    jsonschema.validators.validator_for(schema).check_schema(schema)
                except jsonschema.SchemaError as error:
                    problems.append(f"{where}: invalid JSON Schema ({error.message})")
        names = [c.name for c in case.criteria]
        if len(names) != len(set(names)):
            problems.append(f"{where}: duplicate criteria")
        for criterion in case.criteria:
            try:
                rubric = resolve_rubric(criterion)
            except ValueError as error:
                problems.append(f"{where}: {error}")
                continue
            if rubric.needs_reference and case.reference is None:
                problems.append(f"{where}: criterion {criterion.name} needs a reference answer")
            if rubric.needs_context and not case.context:
                problems.append(f"{where}: criterion {criterion.name} needs retrieved context")
        if not case.checks and not case.criteria and not case.thresholds:
            problems.append(f"{where}: nothing to evaluate (no checks, criteria or thresholds)")
    return problems
