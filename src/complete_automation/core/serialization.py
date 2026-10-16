"""Explicit conversion of framework values to JSON-compatible data.

Unsupported values raise ``TypeError`` instead of being stringified or dropped.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from datetime import date, datetime, time
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

from pydantic import BaseModel, SecretBytes, SecretStr

JSON_INDENT = 2


def to_jsonable(value: Any) -> Any:
    """Convert ``value`` to plain JSON types (dict, list, str, int, float, bool, None)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"non-finite float is not JSON-compatible: {value!r}")
        return value
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (SecretStr, SecretBytes)):
        return "**********"
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            out[key] = to_jsonable(item)
        return out
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable(item) for item in value), key=repr)
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")


def dumps_deterministic(value: Any) -> str:
    """Serialize ``value`` with sorted keys and fixed indentation, ending in a newline."""
    return (
        json.dumps(
            to_jsonable(value),
            sort_keys=True,
            indent=JSON_INDENT,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
