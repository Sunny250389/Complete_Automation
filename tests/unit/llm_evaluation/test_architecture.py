"""The LLM evaluation module depends on the shared core only, executes no model output, and is inert on import."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.cauto_module("llm_evaluation")]

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "complete_automation"
LLM = PACKAGE / "llm_evaluation"
MODULES = sorted(p.relative_to(LLM).as_posix() for p in LLM.rglob("*.py"))
FORBIDDEN = (
    "complete_automation.api", "complete_automation.database", "complete_automation.ui",
    "complete_automation.ml", "complete_automation.llm.", "complete_automation.metrics",
    "complete_automation.utils", "complete_automation.config", "complete_automation.reporting",
    "requests", "aiohttp", "openai", "anthropic", "ollama", "deepeval", "sqlalchemy", "playwright",
    "docker", "kubernetes", "pickle", "marshal", "dill", "subprocess", "socket", "numpy", "pandas", "sklearn",
)  # fmt: skip


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


ALLOWED_LIBRARIES = {"httpx", "jsonschema", "pydantic", "pytest"}  # plus the standard library


def _allowed(name: str) -> bool:
    root = name.split(".")[0]
    if root == "complete_automation":
        return name.startswith(("complete_automation.core", "complete_automation.llm_evaluation"))
    return root in sys.stdlib_module_names or root in ALLOWED_LIBRARIES


@pytest.mark.parametrize("module", MODULES)
def test_imports_only_core_llm_evaluation_stdlib_and_declared_libraries(module: str) -> None:
    for name in _imports(LLM / module):
        assert not name.startswith(FORBIDDEN), f"llm_evaluation/{module} imports {name}"
        assert _allowed(name), f"llm_evaluation/{module} imports {name}, which is not allowed"


def test_model_output_is_never_executed_or_unsafely_deserialised() -> None:
    for module in MODULES:
        tree = ast.parse((LLM / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "compile", "__import__"}, f"{module}: {node.func.id}()"
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"unsafe_load", "full_load"}, f"{module}: {ast.unparse(node)}"
                assert not (node.attr == "load" and ast.unparse(node.value) == "yaml"), f"{module}: yaml.load"


def test_core_and_other_modules_do_not_import_llm_evaluation() -> None:
    for folder in ("core", "api", "database", "ui", "ml"):
        for path in (PACKAGE / folder).rglob("*.py"):
            assert "complete_automation.llm_evaluation" not in path.read_text(encoding="utf-8"), path.name


def test_import_reads_no_files_opens_no_connection_and_configures_no_logging() -> None:
    script = (
        "import logging, socket\n"
        "def refuse(*a, **k): raise AssertionError('network access at import')\n"
        "socket.create_connection = refuse\n"
        "socket.socket.connect = refuse\n"
        "import complete_automation.llm_evaluation as m\n"
        "import complete_automation.llm_evaluation.pytest_plugin\n"
        "assert not logging.getLogger('complete_automation').handlers\n"
        "assert m.OllamaClient() is not None\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


URL_LITERAL = re.compile(r"https?://[^\s\"']+")


def test_the_only_network_target_is_the_default_base_url() -> None:
    # The only URL literal allowed is the default endpoint; every request goes to the configured base_url.
    for module in MODULES:
        urls = set(URL_LITERAL.findall((LLM / module).read_text(encoding="utf-8")))
        assert urls <= {"http://localhost:11434"}, f"{module}: {sorted(urls)}"
