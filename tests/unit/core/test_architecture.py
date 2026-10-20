"""Package-boundary checks for the shared core."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cauto_module("core")

CORE = Path(__file__).resolve().parents[3] / "src" / "complete_automation" / "core"
MODULES = sorted(p.stem for p in CORE.glob("*.py") if p.stem != "__init__")
DOMAIN_PACKAGES = ("deepeval", "openai", "anthropic", "ollama", "httpx", "requests", "sqlalchemy",
                   "playwright", "selenium", "sklearn", "keras", "tensorflow", "torch")


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", MODULES)
def test_core_imports_only_core_and_no_domain_libraries(module: str) -> None:
    for name in imported_modules(CORE / f"{module}.py"):
        if name.startswith("complete_automation"):
            assert name.startswith("complete_automation.core"), f"{module} imports {name}"
        assert name.split(".")[0] not in DOMAIN_PACKAGES, f"{module} imports {name}"


@pytest.mark.parametrize("module", MODULES)
def test_each_module_imports_cleanly_on_its_own(module: str) -> None:
    """A fresh interpreter per module catches circular imports and import-time side effects."""
    script = (
        "import importlib, logging, sys\n"
        f"importlib.import_module('complete_automation.core.{module}')\n"
        "assert not logging.getLogger('complete_automation').handlers, 'logging configured at import'\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in %r]\n" % (DOMAIN_PACKAGES,)
        + "assert not loaded, loaded\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert completed.returncode == 0, completed.stderr
