"""Manage report directories for pytest and evaluation outputs."""

from datetime import datetime, UTC
from pathlib import Path


def create_run_directory(base_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir