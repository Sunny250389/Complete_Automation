"""The database fixtures inside real pytest runs: isolation, cleanup after failures, and shared-core reporting."""

import json
import os
from pathlib import Path

import pytest

from complete_automation.database import EngineManager
from complete_automation.database.sandbox import sqlite_sandbox_settings

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]

CONFTEST = 'pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.database.pytest_plugin"]\n'


def _project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, yaml_body: str) -> Path:
    for name in ("CAUTO_ENVIRONMENT", "CAUTO_EXECUTION__RUN_ID", *(n for n in os.environ if n.startswith("CAUTO_MODULES__DATABASE__"))):
        monkeypatch.delenv(name, raising=False)  # the sub-run must not inherit the caller's database settings
    config = pytester.path / "config"
    (config / "environments").mkdir(parents=True)
    (config / "environments" / "dev.yaml").write_text(yaml_body, encoding="utf-8")
    monkeypatch.setenv("CAUTO_CONFIG_DIR", str(config))
    pytester.makeconftest(CONFTEST)
    return pytester.path


def _results(pytester: pytest.Pytester) -> dict[str, dict]:  # type: ignore[type-arg]
    report = next((pytester.path / "reports").glob("*/results.json"))
    return {r["name"]: r for r in json.loads(report.read_text(encoding="utf-8"))["results"]}


def test_configured_database_sessions_are_isolated_cleaned_up_and_reported(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(
        pytester,
        monkeypatch,
        "modules:\n  database:\n    dialect: sqlite\n    database: data/app.db\n    access: read_write\n",
    )
    (root / "data").mkdir()
    database = root / "data" / "app.db"
    with EngineManager(sqlite_sandbox_settings(database)) as seed, seed.session() as db:
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        db.commit()
    pytester.makepyfile(
        """
        import pytest
        from complete_automation.database import DataAssertionError
        from complete_automation.database.validations import assert_row_count, count_rows

        pytestmark = pytest.mark.cauto_module("database")

        def test_a_writes_without_committing(db_session):
            db_session.execute("INSERT INTO items (name) VALUES ('a')")
            assert count_rows(db_session, "items") == 1

        def test_b_sees_nothing_from_test_a(db_session):
            assert count_rows(db_session, "items") == 0
            db_session.execute("INSERT INTO items (name) VALUES ('b')")

        def test_c_assertion_failure_leaves_no_data(db_session):
            db_session.execute("INSERT INTO items (name) VALUES ('c')")
            assert_row_count(db_session, "items", 99)

        def test_d_unsafe_statement_is_refused(db_session):
            db_session.execute("DROP TABLE items")

        def test_e_bad_sql_is_an_error_not_an_assertion(db_session):
            db_session.query("SELECT * FROM missing_table")

        def test_f_nothing_persisted(db_session):
            assert count_rows(db_session, "items") == 0

        def test_g_the_readonly_manager_refuses_writes_in_the_engine(db_readonly_manager):
            import sqlalchemy
            with db_readonly_manager.session() as db:
                with pytest.raises(sqlalchemy.exc.OperationalError, match="readonly"):
                    db.connection.exec_driver_sql("INSERT INTO items (name) VALUES ('raw')")
        """
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report")

    outcomes = {name: r["status"] for name, r in _results(pytester).items()}
    assert outcomes == {
        "test_a_writes_without_committing": "passed",
        "test_b_sees_nothing_from_test_a": "passed",
        "test_c_assertion_failure_leaves_no_data": "failed",
        "test_d_unsafe_statement_is_refused": "error",
        "test_e_bad_sql_is_an_error_not_an_assertion": "error",
        "test_f_nothing_persisted": "passed",
        "test_g_the_readonly_manager_refuses_writes_in_the_engine": "passed",
    }, result.stdout.str()

    results = _results(pytester)
    failed = results["test_c_assertion_failure_leaves_no_data"]
    assert failed["failure"]["category"] == "assertion" and "expected 99, found 1" in failed["failure"]["message"]
    assert results["test_d_unsafe_statement_is_refused"]["failure"]["category"] == "configuration"
    assert results["test_e_bad_sql_is_an_error_not_an_assertion"]["failure"]["category"] == "unknown"

    # statements and connection are recorded through the shared result contract, without parameters or paths
    passed = results["test_a_writes_without_committing"]
    assert [q["verb"] for q in passed["metadata"]["db_queries"]] == ["INSERT", "SELECT"]
    assert passed["metadata"]["db_connections"] == [
        {"dialect": "sqlite", "database": "app.db", "environment": "dev", "read_only": False}
    ]
    assert {m["name"] for m in passed["metrics"]} == {"db.query_ms"}
    assert [t["outcome"] for t in passed["metadata"]["db_transactions"]] == ["rolled_back"]
    assert passed["artifacts"] == []  # diagnostics are stored only for failing tests by default

    # a failing test gets a sanitized statement history as a shared artifact
    artifact = failed["artifacts"][0]
    stored = next((pytester.path / "artifacts").glob(f"*/{artifact['path']}")).read_text(encoding="utf-8")
    assert artifact["path"] == "database/queries.json"
    assert [q["verb"] for q in json.loads(stored)["queries"]] == ["INSERT", "SELECT"]

    # nothing was committed by any test
    with EngineManager(sqlite_sandbox_settings(database)) as check, check.session() as db:
        assert db.scalar("SELECT count(*) FROM items") == 0
    assert "'a'" not in json.dumps(passed["metadata"]) and "'a'" not in stored  # inline literals never reach reports
    assert str(root) not in json.dumps(passed["metadata"]) + stored  # nor does the database location


def test_sandbox_fixtures_clean_up_after_failures(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(pytester, monkeypatch, "logging:\n  level: INFO\n")
    pytester.makepyfile(
        """
        import pytest
        from complete_automation.database.validations import assert_row_count

        pytestmark = pytest.mark.cauto_module("database")
        STATE = {}

        def test_1_failing_test_with_an_open_transaction(example_session, example_database):
            STATE["manager"] = example_database
            STATE["connection"] = example_session.connection
            example_session.execute("UPDATE customers SET name = 'dirty' WHERE id = 1")
            assert_row_count(example_session, "customers", 99)

        def test_2_everything_was_released(example_session):
            manager = STATE["manager"]
            assert manager.open_connections == 0
            assert manager._engine is None  # the fixture disposed the engine
            assert STATE["connection"].closed
            # this test has its own fresh database: the dirty, uncommitted update is gone
            assert example_session.scalar("SELECT name FROM customers WHERE id = 1") == "Customer 1"

        def test_3_memory_sandbox(sqlite_memory_database):
            with sqlite_memory_database.session() as db:
                db.execute("CREATE TABLE t (a INTEGER)")
                db.commit()
        """
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=2, failed=1)


def test_unconfigured_databases_skip_instead_of_failing(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(pytester, monkeypatch, "logging:\n  level: INFO\n")
    pytester.makepyfile(
        """
        def test_configured_database(db_session):
            raise AssertionError("must be skipped")

        def test_postgres(postgres_session):
            raise AssertionError("must be skipped")
        """
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs")

    result.assert_outcomes(skipped=2)
    result.stdout.fnmatch_lines(["*modules.database is not configured*", "*PostgreSQL is not configured*"])


def test_postgresql_sandbox_needs_explicit_destructive_authorization(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    _project(
        pytester,
        monkeypatch,
        "modules:\n  database:\n    dialect: postgresql\n    host: 127.0.0.1\n    port: 1\n    database: app\n    username: u\n",
    )
    pytester.makepyfile(
        """
        def test_postgres(postgres_session):
            raise AssertionError("must be skipped: the configuration does not authorize a scratch schema")
        """
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*allow_ddl, allow_destructive*privileged_databases*"])


def test_a_configured_but_unreachable_postgresql_server_is_an_error_not_a_skip(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    _project(
        pytester,
        monkeypatch,
        "modules:\n  database:\n    dialect: postgresql\n    host: 127.0.0.1\n    port: 1\n    database: app\n    username: u\n"
        "    connect_timeout_seconds: 1\n    access: read_write\n    allow_ddl: true\n    allow_destructive: true\n"
        "    privileged_databases: [app]\n",
    )
    monkeypatch.setenv("CAUTO_MODULES__DATABASE__PASSWORD", "hunter2")
    pytester.makepyfile(
        """
        def test_postgres(postgres_session):
            raise AssertionError("never reached")
        """
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")

    result.assert_outcomes(errors=1)
    assert "hunter2" not in result.stdout.str() + result.stderr.str()
    result.stdout.fnmatch_lines(["*database connection failed*"])
