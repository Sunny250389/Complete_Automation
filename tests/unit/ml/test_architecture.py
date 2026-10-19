"""The ML module depends on the shared core only, trains nothing, loads no model files and opens no connections."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "complete_automation"
ML = PACKAGE / "ml"
MODULES = sorted(p.relative_to(ML).as_posix() for p in ML.rglob("*.py"))
FORBIDDEN = (
    "complete_automation.api", "complete_automation.database", "complete_automation.ui",
    "complete_automation.llm", "complete_automation.metrics", "complete_automation.utils",
    "complete_automation.config", "complete_automation.reporting",
    "httpx", "requests", "playwright", "selenium", "sqlalchemy", "deepeval", "openai", "ollama",
    "anthropic", "torch", "tensorflow", "keras", "docker", "kubernetes",
    "pickle", "joblib", "dill", "cloudpickle", "matplotlib", "socket",
)  # fmt: skip


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("module", MODULES)
def test_ml_imports_only_core_and_ml(module: str) -> None:
    for name in _imports(ML / module):
        assert not name.startswith(FORBIDDEN), f"ml/{module} imports {name}"
        if name.startswith("complete_automation."):
            assert name.startswith(("complete_automation.core", "complete_automation.ml")), name


def test_no_model_fitting_or_unsafe_deserialisation() -> None:
    for module in MODULES:
        source = (ML / module).read_text(encoding="utf-8")
        for call in (".fit(", ".fit_predict(", ".partial_fit(", "pickle.load", "joblib.load", "eval(", "exec("):
            assert call not in source, f"ml/{module} contains {call}"


def test_core_and_other_modules_do_not_import_ml_or_ml_libraries() -> None:
    for folder in ("core", "api", "database", "ui"):
        for path in (PACKAGE / folder).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "complete_automation.ml" not in source, path.name
            assert not {"sklearn", "pandas", "numpy"} & {n.split(".")[0] for n in _imports(path)}, path.name


def test_import_reads_no_files_opens_no_connection_and_configures_no_logging() -> None:
    script = (
        "import logging, socket, builtins\n"
        "def refuse(*a, **k): raise AssertionError('network access at import')\n"
        "socket.create_connection = refuse\n"
        "socket.socket.connect = refuse\n"
        "import complete_automation.ml as ml\n"
        "import complete_automation.ml.pytest_plugin, complete_automation.ml.fixtures, complete_automation.ml.reporting\n"
        "assert not logging.getLogger('complete_automation').handlers\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
