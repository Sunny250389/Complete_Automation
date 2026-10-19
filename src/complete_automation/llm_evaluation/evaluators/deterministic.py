"""Deterministic checks on a captured response. No judge model, no network, no embeddings.

Each check returns a :class:`MetricOutcome` (passed / failed / error) with an explanation that says
what was expected and what was found. ``error`` means the check could not be evaluated (for
example the case has no reference, or the JSON Schema is itself invalid); it is never a pass.

Transformations applied, and only these:

* ``exact_match``: none by default. ``normalize_whitespace`` strips the ends and collapses runs of
  whitespace; ``case_sensitive=False`` compares with ``str.casefold``. Nothing else (no
  punctuation removal, no stemming, no Unicode normalisation).
* ``contains`` / ``not_contains``: substring search, case-sensitive unless ``case_sensitive=False``.
* ``regex``: ``re.search`` with the pattern as written (``case_sensitive=False`` adds ``re.I``).

Limits: ``refusal`` is a phrase heuristic. It misses refusals worded differently and flags answers
that merely quote a refusal phrase; it is not a safety classifier. ``context_overlap`` is a lexical
proxy for grounding (shared words), not a check that claims are supported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from complete_automation.llm_evaluation.schemas import (
    CheckSpec,
    MetricOutcome,
    ModelResponse,
    Outcome,
    TestCase,
)

try:  # jsonschema belongs to the "llm" (and "api") extra
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the extra
    jsonschema = None  # type: ignore[assignment]

DEFAULT_REFUSAL_PATTERNS = (
    r"\bI (?:can(?:no|')t|cannot|am unable to|won't|will not) (?:help|assist|comply|provide|do that)",
    r"\bI(?:'m| am) (?:sorry|unable)[^.]{0,40}\b(?:can(?:no|')t|cannot|unable|won't)",
    r"\bI (?:must|have to) decline\b",
    r"\bI(?:'m| am) not able to (?:help|assist|provide)",
)

_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it its this that these those "
    "as at by from not no yes can could should would will may might do does did has have had "
    "which who whom what when where why how than then so such into over under about".split()
)
_WORD = re.compile(r"[A-Za-z0-9']+")


class CheckError(Exception):
    """The check cannot be evaluated for this case (reported as outcome ``error``)."""


def run_check(spec: CheckSpec, case: TestCase, response: ModelResponse) -> MetricOutcome:
    handler = _HANDLERS[spec.type]
    try:
        passed, explanation, value, details = handler(spec, case, response)
    except CheckError as error:
        return MetricOutcome(
            name=spec.label, kind="deterministic", outcome=Outcome.ERROR, explanation=str(error)
        )
    return MetricOutcome(
        name=spec.label,
        kind="deterministic",
        outcome=Outcome.PASSED if passed else Outcome.FAILED,
        value=value,
        explanation=explanation,
        details=details,
    )


_Result = tuple[bool, str, float | None, dict[str, Any]]


def _norm(text: str, spec: CheckSpec, *, whitespace: bool = False) -> str:
    if whitespace:
        text = " ".join(text.split())
    return text if spec.case_sensitive else text.casefold()


def _quote(text: str, limit: int = 80) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _not_empty(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    ok = bool(response.text.strip())
    return ok, "response has text" if ok else "response is empty or whitespace only", None, {}


def _exact(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    expected = spec.value if spec.value is not None else case.reference
    if not isinstance(expected, str):
        raise CheckError("exact_match needs a string 'value' or a case reference")
    got = _norm(response.text, spec, whitespace=spec.normalize_whitespace)
    want = _norm(expected, spec, whitespace=spec.normalize_whitespace)
    if got == want:
        return True, "response equals the expected text", None, {}
    return False, f"expected {_quote(expected)}, got {_quote(response.text)}", None, {}


def _needles(spec: CheckSpec) -> list[str]:
    value = spec.value
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not items or not all(isinstance(i, str) and i for i in items):
        raise CheckError(f"{spec.type} needs a non-empty string or list of strings as 'value'")
    return list(items)


def _contains(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    haystack = _norm(response.text, spec)
    missing = [n for n in _needles(spec) if _norm(n, spec) not in haystack]
    if missing:
        return False, f"missing required text: {', '.join(map(_quote, missing))}", None, {"missing": missing}
    return True, "all required text is present", None, {}


def _not_contains(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    haystack = _norm(response.text, spec)
    found = [n for n in _needles(spec) if _norm(n, spec) in haystack]
    if found:
        return False, f"forbidden text present: {', '.join(map(_quote, found))}", None, {"found": found}
    return True, "no forbidden text present", None, {}


def _regex(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    if not isinstance(spec.value, str) or not spec.value:
        raise CheckError("regex needs a pattern as 'value'")
    try:
        pattern = re.compile(spec.value, 0 if spec.case_sensitive else re.IGNORECASE)
    except re.error as error:
        raise CheckError(f"invalid regular expression: {error}") from None
    if pattern.search(response.text):
        return True, f"pattern {_quote(spec.value)} matches", None, {}
    return False, f"pattern {_quote(spec.value)} does not match the response", None, {}


def parse_json(text: str) -> tuple[Any, str | None]:
    """``(value, None)`` or ``(None, reason)``. Accepts one surrounding Markdown code fence."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    if not candidate:
        return None, "response is empty"
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as error:
        return None, f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"


def _json_valid(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    _, problem = parse_json(response.text)
    return problem is None, problem or "response is valid JSON", None, {}


def _json_schema(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    schema = spec.value if spec.value is not None else case.expected_schema
    if not isinstance(schema, dict):
        raise CheckError("json_schema needs a schema as 'value' or the case 'expected_schema'")
    if jsonschema is None:
        raise CheckError("json_schema needs the jsonschema package (install the 'llm' extra)")
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise CheckError(f"the expected schema is itself invalid: {error.message}") from None
    value, problem = parse_json(response.text)
    if problem is not None:
        return False, f"not valid JSON, so the schema was not applied: {problem}", None, {"invalid_json": True}
    errors = sorted(validator_class(schema).iter_errors(value), key=lambda e: list(map(str, e.path)))
    if not errors:
        return True, "JSON conforms to the schema", None, {}
    shown = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]]
    return False, "JSON violates the schema: " + "; ".join(shown), None, {"violations": len(errors)}


def _json_keys(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    required = spec.value
    if not isinstance(required, (list, tuple)) or not all(isinstance(k, str) for k in required):
        raise CheckError("json_keys needs a list of required key names as 'value'")
    value, problem = parse_json(response.text)
    if problem is not None:
        return False, f"not valid JSON: {problem}", None, {"invalid_json": True}
    if not isinstance(value, dict):
        return False, f"expected a JSON object, got {type(value).__name__}", None, {}
    missing = [k for k in required if k not in value]
    unexpected = (
        [k for k in value if k not in set(required) | set(spec.allowed_keys)]
        if spec.allowed_keys is not None
        else []
    )
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected keys: {', '.join(unexpected)}")
        return False, "; ".join(parts), None, {"missing": missing, "unexpected": unexpected}
    return True, "all required keys present", None, {}


def _length(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    if spec.minimum is None and spec.maximum is None:
        raise CheckError("length needs 'minimum' and/or 'maximum'")
    size = len(response.text.split()) if spec.unit == "words" else len(response.text)
    if spec.minimum is not None and size < spec.minimum:
        return False, f"{size} {spec.unit} is below the minimum {spec.minimum:g}", float(size), {}
    if spec.maximum is not None and size > spec.maximum:
        return False, f"{size} {spec.unit} exceeds the maximum {spec.maximum:g}", float(size), {}
    return True, f"{size} {spec.unit} is within the limits", float(size), {}


def _refusal(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    if not isinstance(spec.value, bool):
        raise CheckError("refusal needs 'value': true (must refuse) or false (must not refuse)")
    patterns = spec.patterns or DEFAULT_REFUSAL_PATTERNS
    try:
        refused = any(re.search(p, response.text, re.IGNORECASE) for p in patterns)
    except re.error as error:
        raise CheckError(f"invalid refusal pattern: {error}") from None
    note = " (keyword heuristic)"
    if refused == spec.value:
        return True, ("response refuses" if refused else "response does not refuse") + note, None, {}
    return (
        False,
        ("expected a refusal but none was detected" if spec.value else "response looks like a refusal")
        + note,
        None,
        {},
    )


def _completed(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    if not response.ok:
        return False, f"request status is {response.status.value}", None, {}
    if response.finish_reason == "length":
        return False, "generation stopped at the token limit (finish reason 'length'); output is truncated", None, {}
    if response.finish_reason is None:
        return True, "completed; the provider reported no finish reason, truncation cannot be ruled out", None, {}
    return True, f"completed (finish reason {response.finish_reason!r})", None, {}


def _tokens(text: str) -> set[str]:
    return {w for w in (m.group(0).lower() for m in _WORD.finditer(text)) if len(w) > 2 and w not in _STOPWORDS}


def _context_overlap(spec: CheckSpec, case: TestCase, response: ModelResponse) -> _Result:
    if not case.context:
        raise CheckError("context_overlap needs retrieved context on the test case")
    if spec.minimum is None:
        raise CheckError("context_overlap needs an explicit 'minimum' ratio between 0 and 1")
    answer = _tokens(response.text)
    if not answer:
        return False, "response has no content words to compare", 0.0, {}
    ratio = len(answer & _tokens(" ".join(case.context))) / len(answer)
    ok = ratio >= spec.minimum
    return (
        ok,
        f"{ratio:.2f} of the response's content words appear in the context "
        f"(minimum {spec.minimum:g}); lexical proxy, not a groundedness verdict",
        ratio,
        {},
    )


_HANDLERS: dict[str, Callable[[CheckSpec, TestCase, ModelResponse], _Result]] = {
    "not_empty": _not_empty,
    "exact_match": _exact,
    "contains": _contains,
    "not_contains": _not_contains,
    "regex": _regex,
    "json_valid": _json_valid,
    "json_schema": _json_schema,
    "json_keys": _json_keys,
    "length": _length,
    "refusal": _refusal,
    "completed": _completed,
    "context_overlap": _context_overlap,
}
