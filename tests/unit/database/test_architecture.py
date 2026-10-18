"""The database module depends only on the shared core and its own libraries, and importing it connects to nothing."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "complete_automation"
DATABASE = PACKAGE / "database"
MODULES = sorted(p.relative_to(DATABASE).as_posix() for p in DATABASE.rglob("*.py"))
FORBIDDEN = (
    "complete_automation.api", "complete_automation.llm", "complete_automation.metrics",
    "complete_automation.utils", "complete_automation.config", "complete_automation.reporting",
    "httpx", "playwright", "selenium", "sklearn", "keras", "tensorflow", "torch", "deepeval",
    "openai", "ollama", "requests", "docker", "kubernetes",
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
def test_database_imports_only_core_and_database(module: str) -> None:
    for name in _imports(DATABASE / module):
        assert not name.startswith(FORBIDDEN), f"database/{module} imports {name}"
        if name.startswith("complete_automation."):
            assert name.startswith(("complete_automation.core", "complete_automation.database")), name


def test_the_postgresql_driver_is_imported_nowhere() -> None:
    for module in MODULES:
        assert not any(n.split(".")[0] in ("psycopg", "psycopg2", "asyncpg") for n in _imports(DATABASE / module)), module


def test_core_and_api_do_not_import_the_database_module_or_sqlalchemy() -> None:
    for folder in ("core", "api"):
        for path in (PACKAGE / folder).glob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "complete_automation.database" not in source, path.name
            assert "sqlalchemy" not in source, path.name


def test_no_sql_execution_in_the_shared_core() -> None:
    for path in (PACKAGE / "core").glob("*.py"):
        assert not {"sqlite3", "psycopg", "sqlalchemy"} & {n.split(".")[0] for n in _imports(path)}, path.name


def test_import_opens_no_database_connection() -> None:
    script = (
        "import socket, sqlite3\n"
        "def refuse(*a, **k): raise AssertionError('connection opened at import')\n"
        "socket.create_connection = refuse\n"
        "socket.socket.connect = refuse\n"
        "sqlite3.connect = refuse\n"
        "import sqlalchemy\n"
        "from sqlalchemy.engine import create\n"
        "import complete_automation.database as d\n"
        "import complete_automation.database.pytest_plugin, complete_automation.database.example_schema\n"
        "import complete_automation.database.validations, complete_automation.database.reconciliation\n"
        "import complete_automation.database.sandbox, complete_automation.database.reporting\n"
        "from complete_automation.database.sandbox import sqlite_sandbox_settings\n"
        "m = d.EngineManager(sqlite_sandbox_settings(':memory:'))\n"  # constructing a manager must not connect either
        "m.dispose()\n"
    )

    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)

    assert completed.returncode == 0, completed.stderr


def test_the_postgresql_adapter_is_importable_without_the_driver() -> None:
    script = (
        "import sys\n"
        "sys.modules['psycopg'] = None\n"  # makes `import psycopg` raise ImportError
        "import complete_automation.database.adapters.postgresql as p\n"
        "from complete_automation.database.settings import DatabaseSettings\n"
        "a = p.PostgreSqlAdapter(DatabaseSettings(dialect='postgresql', database='d'), read_only=False)\n"
        "try:\n"
        "    a.validate()\n"
        "except Exception as e:\n"
        "    print(type(e).__name__, e)\n"
    )

    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)

    assert completed.returncode == 0, completed.stderr
    assert "DatabaseConfigurationError" in completed.stdout and "uv sync --extra postgresql" in completed.stdout
