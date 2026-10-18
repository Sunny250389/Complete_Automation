"""The UI module depends only on the shared core and Playwright, and importing it starts nothing."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from complete_automation.core.exceptions import ErrorCategory, classify_exception
from complete_automation.ui import BrowserUnavailableError, UiAssertionError, UiTimeoutError

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]

UI = Path(__file__).resolve().parents[3] / "src" / "complete_automation" / "ui"
FILES = sorted(p for p in UI.rglob("*.py") if p.name != "__init__.py")
FORBIDDEN = ("complete_automation.llm", "complete_automation.metrics", "complete_automation.utils",
             "complete_automation.config", "complete_automation.reporting", "complete_automation.api",
             "complete_automation.database", "selenium", "httpx", "sqlalchemy", "sklearn", "keras",
             "tensorflow", "torch", "deepeval", "openai", "ollama", "requests")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(UI).as_posix())
def test_ui_imports_only_core_and_ui(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            assert not name.startswith(FORBIDDEN), f"{path.name} imports {name}"
            if name.startswith("complete_automation."):
                assert name.startswith(("complete_automation.core", "complete_automation.ui")), name


def test_core_does_not_import_the_ui_module() -> None:
    for path in (UI.parent / "core").glob("*.py"):
        assert "complete_automation.ui" not in path.read_text(encoding="utf-8"), path.name


def test_importing_the_package_needs_no_playwright() -> None:
    script = (
        "import sys\n"
        "sys.modules['playwright'] = None  # any 'import playwright' now fails\n"
        "import complete_automation.ui\n"
        "from complete_automation.ui import UiSettings, UiAssertionError\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert completed.returncode == 0, completed.stderr


def test_import_launches_no_browser_and_opens_no_connection() -> None:
    script = (
        "import socket, subprocess\n"
        "import playwright.sync_api  # asyncio subclasses Popen at import: load it before patching\n"
        "def refuse(*a, **k): raise AssertionError('side effect at import')\n"
        "socket.create_connection = refuse\n"
        "socket.socket.connect = refuse\n"
        "subprocess.Popen = refuse\n"
        "import complete_automation.ui.browser, complete_automation.ui.pages, complete_automation.ui.assertions\n"
        "import complete_automation.ui.diagnostics, complete_automation.ui.pytest_plugin\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert completed.returncode == 0, completed.stderr


def test_errors_are_classified_through_the_shared_core() -> None:
    assert classify_exception(UiAssertionError("x")) is ErrorCategory.ASSERTION
    assert isinstance(UiAssertionError("x"), AssertionError)
    assert classify_exception(UiTimeoutError("x")) is ErrorCategory.TIMEOUT
    assert isinstance(UiTimeoutError("x"), TimeoutError)
    assert classify_exception(BrowserUnavailableError("x")) is ErrorCategory.CONFIGURATION
