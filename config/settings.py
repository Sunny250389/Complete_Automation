"""Centralized settings for pytest + DeepEval test automation."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "llm_automation" / "data" / "testdata"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "gpt-4.1")
CONFIDENT_ALIAS = os.getenv("CONFIDENT_ALIAS", "llm-regression-suite")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")