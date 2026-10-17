"""JSON Schema validation of API payloads (``jsonschema``; the dialect comes from ``$schema``).

Formats such as ``email`` and ``date-time`` are checked. Violation messages
name the failing location and keyword; values are redacted and truncated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for

from complete_automation.api.client import ApiResponse
from complete_automation.api.errors import ContractError
from complete_automation.core.datasets import load_json, load_yaml
from complete_automation.core.exceptions import TestDataError
from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor

MAX_REPORTED = 20


@dataclass(frozen=True)
class SchemaViolation:
    location: str
    message: str
    keyword: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


def json_location(prefix: str, path: Iterable[Any]) -> str:
    text = prefix
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text


def describe_error(error: ValidationError, prefix: str, redactor: Redactor) -> SchemaViolation:
    location = json_location(prefix, error.absolute_path)
    sensitive = any(isinstance(p, str) and redactor.is_sensitive_key(p) for p in error.absolute_path)
    if sensitive:
        message = f"value fails '{error.validator}'"
    else:
        message = redactor.redact_text(error.message)
        if len(message) > 200:
            message = message[:197] + "..."
    return SchemaViolation(location, message, str(error.validator))


def load_schema(path: Path) -> dict[str, Any]:
    """Load a JSON Schema from a .json or .yaml file and check that it is a valid schema."""
    path = Path(path)
    try:
        schema = load_yaml(path) if path.suffix.lower() in (".yaml", ".yml") else load_json(path)
    except TestDataError as error:
        raise ContractError(f"cannot load JSON Schema {path}: {error}") from error
    check_schema(schema, source=str(path))
    return schema


def check_schema(schema: Any, *, source: str = "schema") -> None:
    if not isinstance(schema, dict):
        raise ContractError(f"{source}: a JSON Schema must be an object")
    try:
        validator_for(schema, default=Draft202012Validator).check_schema(schema)
    except SchemaError as error:
        raise ContractError(f"{source} is not a valid JSON Schema: {error.message}") from None


def schema_violations(
    instance: Any, schema: dict[str, Any], *, label: str = "body", redactor: Redactor | None = None
) -> list[SchemaViolation]:
    """Every way ``instance`` violates ``schema``, sorted by location."""
    check_schema(schema)
    redactor = redactor or DEFAULT_REDACTOR
    validator = validator_for(schema, default=Draft202012Validator)(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda e: [str(p) for p in e.absolute_path])
    return [describe_error(error, label, redactor) for error in errors]


def assert_matches_schema(target: ApiResponse | Any, schema: dict[str, Any], *, label: str = "body") -> None:
    """Fail with every violation (up to 20) when the JSON payload does not match ``schema``."""
    if isinstance(target, ApiResponse):
        data, context, redactor = target.json(), f"{target.describe()}: ", target.sanitizer.redactor
    else:
        data, context, redactor = target, "", DEFAULT_REDACTOR
    violations = schema_violations(data, schema, label=label, redactor=redactor)
    if violations:
        shown = "\n".join(f"  - {v}" for v in violations[:MAX_REPORTED])
        more = f"\n  ... {len(violations) - MAX_REPORTED} more" if len(violations) > MAX_REPORTED else ""
        raise AssertionError(f"{context}{label} does not match the schema ({len(violations)} violation(s)):\n{shown}{more}")
