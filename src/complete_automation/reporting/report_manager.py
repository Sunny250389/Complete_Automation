"""Manage report directories for pytest and evaluation outputs.

Kept for existing callers. New code should use
``complete_automation.core.artifacts.LocalArtifactStore`` and
``complete_automation.core.reporting.JsonReporter``.
"""

from datetime import UTC, datetime
from pathlib import Path


def create_run_directory(base_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
