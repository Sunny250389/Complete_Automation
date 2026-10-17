"""API assertions with diagnostics: operation, method, expectation and a sanitized excerpt.

Every helper takes an ``ApiResponse``; JSON helpers also accept already
decoded data. Values under sensitive keys are masked in messages.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any

from complete_automation.api.client import ApiResponse
from complete_automation.core.assertions import assert_subset
from complete_automation.core.redaction import DEFAULT_REDACTOR

_MISSING: Any = object()
_TOKEN = re.compile(r"\.?([^.\[\]]+)|\[(-?\d+)\]")
_EXCERPT = 300


def _fail(response: ApiResponse | None, condition: str, *, excerpt: bool = True) -> AssertionError:
    if response is None:
        return AssertionError(condition)
    message = f"{response.describe()}: {condition}"
    if excerpt and response.content:
        message += f"\n  body: {response.body_excerpt(_EXCERPT)}"
    return AssertionError(message)


def _show(value: Any, key: str = "") -> str:
    if key and DEFAULT_REDACTOR.is_sensitive_key(key):
        return DEFAULT_REDACTOR.mask
    text = DEFAULT_REDACTOR.redact_text(repr(value))
    return text if len(text) <= 80 else text[:77] + "..."


def _data(target: ApiResponse | Any) -> tuple[Any, ApiResponse | None]:
    return (target.json(), target) if isinstance(target, ApiResponse) else (target, None)


# --------------------------------------------------------------------------- HTTP level


def assert_status(response: ApiResponse, expected: int | Iterable[int]) -> None:
    """Status equals ``expected`` (an int) or is one of ``expected`` (iterable or range)."""
    allowed = {expected} if isinstance(expected, int) else set(expected)
    if response.status_code not in allowed:
        wanted = expected if isinstance(expected, int) else sorted(allowed)
        raise _fail(response, f"expected status {wanted}, got {response.status_code}")


def assert_status_class(response: ApiResponse, klass: int) -> None:
    """Status is in the ``klass`` hundred range, e.g. ``2`` for any 2xx."""
    if response.status_code // 100 != klass:
        raise _fail(response, f"expected a {klass}xx status, got {response.status_code}")


def assert_latency(response: ApiResponse, max_ms: float, *, include_retries: bool = False) -> None:
    """Client-observed latency of the final attempt (or of the whole call) is within ``max_ms``."""
    observed = response.total_ms if include_retries else response.latency_ms
    scope = "total time including retries" if include_retries else "latency"
    if observed > max_ms:
        raise _fail(response, f"{scope} {observed:.1f} ms exceeds {max_ms:.1f} ms", excerpt=False)


def assert_header(response: ApiResponse, name: str, expected: str | re.Pattern[str] | None = None) -> None:
    """Header present and, if given, equal to ``expected`` or matching a compiled pattern."""
    value = response.headers.get(name)
    if value is None:
        raise _fail(response, f"missing header {name!r}", excerpt=False)
    if expected is None:
        return
    ok = expected.search(value) is not None if isinstance(expected, re.Pattern) else value == expected
    if not ok:
        shown = response.sanitizer.mask if response.sanitizer.is_sensitive(name) else _show(value)
        wanted = f"matching /{expected.pattern}/" if isinstance(expected, re.Pattern) else repr(expected)
        raise _fail(response, f"header {name!r} expected {wanted}, got {shown}", excerpt=False)


def assert_headers_present(response: ApiResponse, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in response.headers]
    if missing:
        raise _fail(response, f"missing header(s) {missing}", excerpt=False)


def assert_content_type(response: ApiResponse, expected: str) -> None:
    """Media type (parameters such as charset ignored) equals ``expected``."""
    if response.content_type != expected.lower():
        raise _fail(response, f"expected content type {expected!r}, got {response.content_type!r}")


def assert_redirect(response: ApiResponse, *, location: str | None = None, status: int | None = None) -> None:
    """An unfollowed redirect (3xx with ``Location``), optionally to ``location`` with ``status``."""
    if not response.is_redirect:
        raise _fail(response, f"expected a redirect, got status {response.status_code}")
    if status is not None and response.status_code != status:
        raise _fail(response, f"expected redirect status {status}, got {response.status_code}", excerpt=False)
    target = response.headers.get("Location", "")
    if location is not None and target != location:
        raise _fail(
            response, f"expected Location {location!r}, got {response.sanitizer.url(target)!r}", excerpt=False
        )


def assert_error_response(
    response: ApiResponse, status: int | Iterable[int] | None = None, *, fields: Iterable[str] = ("error",)
) -> None:
    """A 4xx/5xx status (or exactly ``status``) with a JSON error body holding ``fields``."""
    if status is not None:
        assert_status(response, status)
    elif response.status_code < 400:
        raise _fail(response, f"expected an error status (4xx/5xx), got {response.status_code}")
    data = response.json()
    if not isinstance(data, Mapping):
        raise _fail(response, f"expected a JSON object error body, got {type(data).__name__}")
    missing = [field for field in fields if field not in data]
    if missing:
        raise _fail(response, f"error body lacks field(s) {missing}")


# --------------------------------------------------------------------------- payload level


def get_path(data: Any, path: str) -> Any:
    """Value at ``path`` such as ``data[0].email`` or ``[1]``; AssertionError if absent."""
    node = data
    walked = "$"
    for key, index in _TOKEN.findall(path):
        if index:
            if not isinstance(node, list) or not -len(node) <= int(index) < len(node):
                raise AssertionError(f"{walked}[{index}]: no such item")
            node = node[int(index)]
            walked += f"[{index}]"
        else:
            if not isinstance(node, Mapping) or key not in node:
                raise AssertionError(f"{walked}.{key}: missing")
            node = node[key]
            walked += f".{key}"
    return node


def assert_json_keys(
    target: ApiResponse | Any,
    required: Iterable[str] = (),
    *,
    forbidden: Iterable[str] = (),
    allow_extra: bool = True,
    path: str = "",
) -> None:
    """The object at ``path`` has every ``required`` key, no ``forbidden`` key and,
    unless ``allow_extra``, nothing else."""
    data, response = _data(target)
    node = get_path(data, path) if path else data
    where = path or "$"
    if not isinstance(node, Mapping):
        raise _fail(response, f"{where}: expected a JSON object, got {type(node).__name__}")
    required = list(required)
    problems = []
    missing = sorted(set(required) - set(node))
    present_forbidden = sorted(set(forbidden) & set(node))
    unexpected = sorted(set(node) - set(required)) if not allow_extra else []
    if missing:
        problems.append(f"missing {missing}")
    if present_forbidden:
        problems.append(f"forbidden {present_forbidden}")
    if unexpected:
        problems.append(f"unexpected {unexpected}")
    if problems:
        raise _fail(response, f"{where}: keys " + "; ".join(problems))


def assert_json_subset(target: ApiResponse | Any, expected: Any, *, path: str = "") -> None:
    """Every entry of ``expected`` appears, equal, at ``path`` (extra keys allowed)."""
    data, response = _data(target)
    node = get_path(data, path) if path else data
    try:
        assert_subset(node, expected, label=path or "$")
    except AssertionError as error:
        raise _fail(response, str(error), excerpt=False) from None


def assert_field(
    target: ApiResponse | Any,
    path: str,
    *,
    equals: Any = _MISSING,
    type_: type | tuple[type, ...] | None = None,
    nullable: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    pattern: str | None = None,
    one_of: Iterable[Any] | None = None,
    datetime_format: str | None = None,
) -> None:
    """Check one field: value, type (``bool`` is not an ``int``), nullability,
    numeric bounds, regex, allowed values, or a date format (``"iso8601"`` or a
    ``strptime`` format)."""
    data, response = _data(target)
    value = get_path(data, path)
    key = path.rsplit(".", 1)[-1].split("[", 1)[0]

    def fail(condition: str) -> AssertionError:
        return _fail(response, f"{path}: {condition}, got {_show(value, key)}")

    if value is None:
        if equals is None or (nullable and equals is _MISSING):
            return
        raise fail("must not be null")
    if equals is not _MISSING and (value != equals or type(value) is not type(equals)):
        raise fail(f"expected {_show(equals, key)}")
    if type_ is not None:
        kinds = type_ if isinstance(type_, tuple) else (type_,)
        if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
            raise fail(f"expected type {'/'.join(k.__name__ for k in kinds)}")
    if minimum is not None or maximum is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            raise fail("expected a number")
        if minimum is not None and value < minimum:
            raise fail(f"expected >= {minimum}")
        if maximum is not None and value > maximum:
            raise fail(f"expected <= {maximum}")
    if pattern is not None and (not isinstance(value, str) or re.fullmatch(pattern, value) is None):
        raise fail(f"expected to match /{pattern}/")
    if one_of is not None and value not in list(one_of):
        raise fail(f"expected one of {list(one_of)}")
    if datetime_format is not None:
        try:
            if not isinstance(value, str):
                raise ValueError
            if datetime_format == "iso8601":
                datetime.fromisoformat(value)
            else:
                datetime.strptime(value, datetime_format)
        except ValueError:
            raise fail(f"expected a {datetime_format} timestamp") from None


def assert_collection(
    target: ApiResponse | Any,
    path: str = "",
    *,
    min_items: int | None = None,
    max_items: int | None = None,
    contains: Any = _MISSING,
    unique_by: str | None = None,
    each: Callable[[Any], bool] | None = None,
) -> None:
    """Checks on the array at ``path``: size, membership, uniqueness of a field, a predicate per item."""
    data, response = _data(target)
    items = get_path(data, path) if path else data
    where = path or "$"
    if not isinstance(items, list):
        raise _fail(response, f"{where}: expected an array, got {type(items).__name__}")
    if min_items is not None and len(items) < min_items:
        raise _fail(response, f"{where}: expected at least {min_items} item(s), got {len(items)}")
    if max_items is not None and len(items) > max_items:
        raise _fail(response, f"{where}: expected at most {max_items} item(s), got {len(items)}")
    if contains is not _MISSING and contains not in items:
        raise _fail(response, f"{where}: expected to contain {_show(contains)}")
    if unique_by is not None:
        values = [get_path(item, unique_by) for item in items]
        duplicates = sorted({_show(v) for v in values if values.count(v) > 1})
        if duplicates:
            raise _fail(response, f"{where}: '{unique_by}' is not unique (duplicates: {duplicates})")
    if each is not None:
        failing = [index for index, item in enumerate(items) if not each(item)]
        if failing:
            raise _fail(response, f"{where}: item(s) {failing[:10]} fail the check")


def assert_pagination(
    target: ApiResponse | Any,
    *,
    items: str = "data",
    page: str = "page",
    per_page: str = "per_page",
    total: str = "total",
    total_pages: str | None = "total_pages",
) -> None:
    """Pagination metadata is consistent with the page of items it describes."""
    data, response = _data(target)
    values = {}
    for name in filter(None, (page, per_page, total, total_pages)):
        value = get_path(data, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _fail(response, f"{name}: expected a non-negative integer, got {_show(value)}")
        values[name] = value
    rows = get_path(data, items)
    if not isinstance(rows, list):
        raise _fail(response, f"{items}: expected an array")
    size, current, count = values[per_page], values[page], values[total]
    if size < 1 or current < 1:
        raise _fail(response, f"{page} and {per_page} must be at least 1")
    expected_pages = math.ceil(count / size)
    if total_pages and values[total_pages] != expected_pages:
        raise _fail(response, f"{total_pages} is {values[total_pages]}, but {total}/{per_page} gives {expected_pages}")
    expected_rows = max(0, min(size, count - (current - 1) * size))
    if len(rows) != expected_rows:
        raise _fail(response, f"page {current} should hold {expected_rows} item(s), got {len(rows)}")
