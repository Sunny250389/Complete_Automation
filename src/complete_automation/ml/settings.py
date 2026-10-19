"""Settings for the ML module, read from ``modules.ml`` of the shared configuration.

Example (``config/environments/<env>.yaml``)::

    modules:
      ml:
        data_root: data/ml          # datasets are loaded from here, nowhere else
        baseline_dir: data/ml/baselines
        max_rows: 100000
        publish_artifacts: true
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MLSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_root: Path = Path("data/ml")
    baseline_dir: Path = Path("data/ml/baselines")
    max_rows: int = Field(default=100_000, ge=1)
    max_file_bytes: int = Field(default=50_000_000, ge=1)
    """Files are read fully into memory; larger files are refused, not streamed."""
    publish_artifacts: bool = True
    """Store the JSON report and CSV summaries through the shared artifact store."""

    def with_base_dir(self, base_dir: Path) -> MLSettings:
        """Resolve relative paths against ``base_dir`` (the project root)."""
        base = Path(base_dir)
        return self.model_copy(
            update={
                "data_root": _resolve(base, self.data_root),
                "baseline_dir": _resolve(base, self.baseline_dir),
            }
        )


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
