"""Datasets: the in-memory contract and a loader confined to one data root.

Supported formats: CSV and JSON (a list of records), read with the shared-core loaders.
Parquet is not supported (it would need an extra binary dependency); convert to CSV.

Limits: files are read fully into memory, so ``max_file_bytes`` and ``max_rows`` refuse large
inputs instead of streaming them. CSV cells are text; undeclared columns are converted to numbers
when every value parses, declared ``string`` columns stay text, and empty cells become missing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from complete_automation.core.datasets import DataLoader
from complete_automation.ml.errors import DatasetError, MLConfigurationError
from complete_automation.ml.schemas import DatasetMetadata, DatasetSchema

SUPPORTED_SUFFIXES = (".csv", ".json")


@dataclass(frozen=True)
class Dataset:
    """Features, optional target (ground truth) and metadata. Never modified by the evaluator."""

    frame: pd.DataFrame
    metadata: DatasetMetadata
    features: tuple[str, ...]
    target: str | None = None
    schema: DatasetSchema | None = None

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        dataset_id: str,
        *,
        features: tuple[str, ...] | list[str],
        target: str | None = None,
        version: str | None = None,
        schema: DatasetSchema | None = None,
        source: str | None = None,
    ) -> Dataset:
        if not frame.index.is_unique:
            raise DatasetError("dataset index must be unique so predictions can be aligned to rows")
        copy = frame.copy(deep=True)
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            version=version,
            source=source,
            n_rows=len(copy),
            n_columns=copy.shape[1],
            fingerprint=fingerprint(copy),
        )
        return cls(copy, metadata, tuple(features), target, schema)

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional name
        return self.frame[list(self.features)]

    @property
    def y(self) -> pd.Series:
        if self.target is None:
            raise MLConfigurationError("this dataset has no target column")
        return self.frame[self.target]

    def subset(self, keep: np.ndarray) -> Dataset:
        """Rows where ``keep`` is true; the original index is preserved."""
        return Dataset(self.frame[keep], self.metadata, self.features, self.target, self.schema)


def fingerprint(frame: pd.DataFrame) -> str:
    """Identify the data without storing it: hash of column names and per-cell hashes."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, frame.columns)).encode())
    try:
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    except TypeError:  # unhashable cells: fall back to their text form
        digest.update(frame.astype(str).to_csv(index=False).encode())
    return digest.hexdigest()


class DatasetLoader:
    """Loads datasets from files under ``root``; paths cannot escape it."""

    def __init__(
        self, root: Path, *, max_rows: int = 100_000, max_file_bytes: int = 50_000_000
    ) -> None:
        self._loader = DataLoader(root)
        self.root = self._loader.root
        self.max_rows = max_rows
        self.max_file_bytes = max_file_bytes

    def load(
        self,
        relative: str | Path,
        *,
        features: tuple[str, ...] | list[str],
        target: str | None = None,
        dataset_id: str | None = None,
        version: str | None = None,
        schema: DatasetSchema | None = None,
    ) -> Dataset:
        path = self._loader.path(relative)
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise DatasetError(
                f"unsupported dataset format {path.suffix!r}; use one of {SUPPORTED_SUFFIXES}"
            )
        if path.is_file() and path.stat().st_size > self.max_file_bytes:
            raise DatasetError(
                f"dataset file is larger than max_file_bytes={self.max_file_bytes}: {relative}"
            )
        records = self._loader.load(relative)
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise DatasetError(f"{relative}: expected a list of records")
        if len(records) > self.max_rows:
            raise DatasetError(f"{relative}: {len(records)} rows exceed max_rows={self.max_rows}")
        frame = pd.DataFrame.from_records(records) if records else pd.DataFrame()
        if path.suffix.lower() == ".csv":
            frame = _type_csv_columns(frame, schema)
        return Dataset.from_frame(
            frame,
            dataset_id or path.stem,
            features=features,
            target=target,
            version=version,
            schema=schema,
            source=Path(relative).as_posix(),
        )


def _type_csv_columns(frame: pd.DataFrame, schema: DatasetSchema | None) -> pd.DataFrame:
    out = frame.replace("", np.nan)
    for column in out.columns:
        spec = schema.columns.get(column) if schema else None
        if spec is not None and spec.dtype == "string":
            continue
        if spec is not None and spec.dtype == "boolean":
            lowered = out[column].astype("object").map(
                lambda v: v if pd.isna(v) else {"true": True, "false": False}.get(str(v).lower(), v)
            )
            out[column] = lowered
            continue
        try:
            out[column] = pd.to_numeric(out[column])
        except (ValueError, TypeError):
            pass  # stays text; validation reports a declared numeric column that is not
    return out


def column_is(series: pd.Series, dtype: str) -> bool:
    """Whether ``series`` satisfies a :class:`ColumnSpec` dtype."""
    kind = pd.api.types
    non_null = series.dropna()
    if dtype == "any":
        return True
    if dtype == "boolean":
        return bool(kind.is_bool_dtype(series) or non_null.map(lambda v: isinstance(v, bool)).all())
    if dtype == "string":
        return bool(non_null.map(lambda v: isinstance(v, str)).all())
    numeric = kind.is_numeric_dtype(series) and not kind.is_bool_dtype(series)
    if dtype == "numeric":
        return bool(numeric)
    if dtype == "integer":
        return bool(numeric and (kind.is_integer_dtype(series) or (non_null % 1 == 0).all()))
    raise ValueError(dtype)


def is_numeric_column(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def to_python(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
