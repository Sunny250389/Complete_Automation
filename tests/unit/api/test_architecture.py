"""The API module depends only on the shared core and its own libraries, and imports without network access."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]

API = Path(__file__).resolve().parents[3] / "src" / "complete_automation" / "api"
MODULES = sorted(p.stem for p in API.glob("*.py") if p.stem != "__init__")
FORBIDDEN = ("complete_automation.llm", "complete_automation.metrics", "complete_automation.utils",
             "complete_automation.config", "complete_automation.reporting", "playwright", "selenium",
             "sqlalchemy", "sklearn", "keras", "tensorflow", "torch", "deepeval", "openai", "ollama", "requests")


@pytest.mark.parametrize("module", MODULES)
def test_api_imports_only_core_and_api(module: str) -> None:
    tree = ast.parse((API / f"{module}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            assert not name.startswith(FORBIDDEN), f"api/{module}.py imports {name}"
            if name.startswith("complete_automation."):
                assert name.startswith(("complete_automation.core", "complete_automation.api")), name


def test_core_does_not_import_the_api_module() -> None:
    core = API.parent / "core"
    for path in core.glob("*.py"):
        assert "complete_automation.api" not in path.read_text(encoding="utf-8"), path.name


def test_import_performs_no_network_access() -> None:
    script = (
        "import socket\n"
        "def refuse(*a, **k): raise AssertionError('network access at import')\n"
        "socket.create_connection = refuse\n"
        "socket.socket.connect = refuse\n"
        "import complete_automation.api, complete_automation.api.contract, complete_automation.api.pytest_plugin\n"
        "import complete_automation.api.mocking, complete_automation.api.schema, complete_automation.api.assertions\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert completed.returncode == 0, completed.stderr
