"""Assertion helpers for generic structured data (API bodies, DB rows, evaluation output).

Use plain ``assert`` for everything else; these exist because comparing nested
mappings with ``assert`` gives a poor message when only part of the data matters.
Messages show paths and short, redacted values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from complete_automation.core.redaction import DEFAULT_REDACTOR


def assert_keys_present(data: Mapping[str, Any], required: Iterable[str], *, label: str = "data") -> None:
    """Fail when ``data`` lacks any of the ``required`` keys."""
    if not isinstance(data, Mapping):
        raise AssertionError(f"{label}: expected a mapping, got {type(data).__name__}")
    missing = sorted(set(required) - set(data))
    if missing:
        raise AssertionError(
            f"{label}: missing required key(s) {missing}; present: {sorted(map(str, data))}"
        )


def assert_subset(actual: Any, expected: Any, *, label: str = "data") -> None:
    """Fail unless every entry of ``expected`` appears, equal, in ``actual``.

    Mappings match recursively (extra keys in ``actual`` are allowed); lists must
    have equal length and match element by element; other values must be equal.
    All mismatching paths are reported at once.
    """
    problems: list[str] = []
    _compare(actual, expected, label, problems)
    if problems:
        raise AssertionError(f"{label}: {len(problems)} mismatch(es):\n" + "\n".join(problems))


def _compare(actual: Any, expected: Any, path: str, problems: list[str]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            problems.append(f"  - {path}: expected a mapping, got {type(actual).__name__}")
            return
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"  - {path}.{key}: missing")
            else:
                _compare(actual[key], value, f"{path}.{key}", problems)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            problems.append(f"  - {path}: expected a list, got {type(actual).__name__}")
        elif len(actual) != len(expected):
            problems.append(f"  - {path}: expected {len(expected)} item(s), got {len(actual)}")
        else:
            for index, (left, right) in enumerate(zip(actual, expected)):
                _compare(left, right, f"{path}[{index}]", problems)
    elif actual != expected or type(actual) is not type(expected):
        problems.append(f"  - {path}: expected {_show(path, expected)}, got {_show(path, actual)}")


def _show(path: str, value: Any, limit: int = 80) -> str:
    """Short, redacted rendering of a value; values under sensitive keys are masked."""
    last_key = path.rsplit(".", 1)[-1].split("[", 1)[0]
    if DEFAULT_REDACTOR.is_sensitive_key(last_key):
        return DEFAULT_REDACTOR.mask
    text = DEFAULT_REDACTOR.redact_text(repr(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."
