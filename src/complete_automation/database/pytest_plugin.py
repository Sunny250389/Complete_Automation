"""Database fixtures, built on the shared-core fixtures.

Load in the ``conftest.py`` of database test folders::

    pytest.importorskip("sqlalchemy")  # skip cleanly without the "database" extra
    from complete_automation.database.pytest_plugin import *  # noqa: F401,F403

Nothing is autouse and nothing connects while a fixture is being *set up*
except the ones that hand you an open session.

Configured database (``modules.database``; opt-in, skipped when unset)
---------------------------------------------------------------------------

=========================  ========  =====================================================
``database_settings``      session   ``modules.database`` settings, file paths resolved
``database_manager``       session   :class:`EngineManager` for the configured database;
                                     disposed at the end of the session
``db_session_factory``     function  ``factory(manager=None, **options)`` -> :class:`DatabaseSession`;
                                     every session is rolled back and closed after the test
``db_session``             function  a session on the configured database; read-only unless
                                     ``access: read_write``; **never committed unless the test commits**
``db_readonly_manager``    session   a second manager forced to ``access: read_only`` (engine-level)
=========================  ========  =====================================================

Sandboxes the test owns (always available with the ``database`` extra)
--------------------------------------------------------------------------

=========================  ========  =====================================================
``sqlite_database``        function  manager for a fresh temporary SQLite *file*; removed after
                                     the test (engine disposed first, which Windows needs)
``sqlite_memory_database`` function  manager for a fresh ``:memory:`` database (one shared connection)
``example_database``       function  ``sqlite_database`` with the example schema and seed data committed
``example_session``        function  a session on ``example_database``
=========================  ========  =====================================================

PostgreSQL (opt-in; skipped unless explicitly configured)
--------------------------------------------------------------------------

=========================  ========  =====================================================
``postgres_database``      function  manager bound to a scratch schema created for this test and
                                     dropped afterwards; needs ``dialect: postgresql``,
                                     ``access: read_write``, ``allow_ddl``, ``allow_destructive``
                                     and the database in ``privileged_databases``
``postgres_session``       function  a session on that scratch schema, example schema created
=========================  ========  =====================================================

Sessions record their statements in the test's shared ``TestResult``
(``metadata.db_queries``, ``db.query_ms`` metrics) and, per
``modules.database.capture_diagnostics`` (default: on failure), store a
sanitized statement history as an artifact. Override any fixture by defining
one with the same name in a closer ``conftest.py``.

Fixture scope and isolation: managers for the configured database are
session-scoped (one pool for the run); sessions are function-scoped, so no
transaction or connection is shared between tests. Sandboxes are created per
test, so tests cannot contaminate each other and may commit freely.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.config import FrameworkSettings
from complete_automation.core.context import RunContext
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.database.engine import EngineManager
from complete_automation.database.errors import DatabaseError
from complete_automation.database.example_schema import create_example_schema, seed_example_data
from complete_automation.database.reporting import (
    record_connection,
    record_query,
    record_transactions,
    save_diagnostics,
)
from complete_automation.database.sandbox import sqlite_sandbox_settings, unique_schema_name
from complete_automation.database.session import DatabaseSession
from complete_automation.database.settings import MEMORY, DatabaseSettings

__all__ = [
    "database_manager",
    "database_settings",
    "db_readonly_manager",
    "db_session",
    "db_session_factory",
    "example_database",
    "example_session",
    "postgres_database",
    "postgres_session",
    "pytest_runtest_makereport",
    "sqlite_database",
    "sqlite_memory_database",
]

_FAILED = pytest.StashKey[bool]()
_NOT_CONFIGURED = (
    "modules.database is not configured (set it in the environment file or with "
    "CAUTO_MODULES__DATABASE__DIALECT / __DATABASE / ...)"
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    report = yield
    if report.when in ("setup", "call") and report.failed:
        item.stash[_FAILED] = True
    return report


# --- configured database ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def database_settings(framework_settings: FrameworkSettings) -> DatabaseSettings:
    return framework_settings.section("database", DatabaseSettings).with_base_dir(framework_settings.base_dir)


@pytest.fixture(scope="session")
def database_manager(
    database_settings: DatabaseSettings, framework_settings: FrameworkSettings, run_context: RunContext
) -> Iterator[EngineManager]:
    if not database_settings.is_configured:
        pytest.skip(_NOT_CONFIGURED)
    manager = EngineManager(database_settings, environment=run_context.environment, redactor=framework_settings.redactor())
    try:
        yield manager
    finally:
        manager.dispose()


@pytest.fixture(scope="session")
def db_readonly_manager(
    database_settings: DatabaseSettings, framework_settings: FrameworkSettings, run_context: RunContext
) -> Iterator[EngineManager]:
    """A manager for the configured database whose connections the *engine* makes read-only."""
    if not database_settings.is_configured:
        pytest.skip(_NOT_CONFIGURED)
    restricted = database_settings.model_copy(
        update={"access": "read_only", "allow_ddl": False, "allow_destructive": False, "privileged_databases": ()}
    )
    manager = EngineManager(restricted, environment=run_context.environment, redactor=framework_settings.redactor())
    try:
        yield manager
    finally:
        manager.dispose()


@pytest.fixture
def db_session_factory(
    request: pytest.FixtureRequest,
    database_settings: DatabaseSettings,
    result_recorder: ResultRecorder,
    artifact_store: LocalArtifactStore,
) -> Iterator[Callable[..., DatabaseSession]]:
    """``factory(manager, **session_options)`` -> a session that is cleaned up after the test."""
    opened: list[DatabaseSession] = []

    def factory(manager: EngineManager, **options: Any) -> DatabaseSession:
        options.setdefault("on_query", lambda summary: record_query(result_recorder, summary))
        session = manager.session(**options)
        opened.append(session)
        record_connection(result_recorder, manager)
        return session

    try:
        yield factory
    finally:
        failed = request.node.stash.get(_FAILED, False)
        problems: list[Exception] = []
        for session in opened:
            try:
                session.close()  # first: the closing rollback is itself an outcome worth reporting
            except DatabaseError as error:
                problems.append(error)
            try:
                if _wants_diagnostics(database_settings, failed):
                    result_recorder.attach(
                        save_diagnostics(artifact_store, session.manager, session.history, session.transaction_outcomes)
                    )
                record_transactions(result_recorder, session.transaction_outcomes)
            except Exception as error:  # reporting must not hide cleanup  # noqa: BLE001
                problems.append(error)
        if problems:
            raise problems[0]


def _wants_diagnostics(settings: DatabaseSettings, failed: bool) -> bool:
    return settings.capture_diagnostics == "always" or (settings.capture_diagnostics == "on_failure" and failed)


@pytest.fixture
def db_session(database_manager: EngineManager, db_session_factory: Callable[..., DatabaseSession]) -> DatabaseSession:
    return db_session_factory(database_manager)


# --- sandboxes ----------------------------------------------------------------------------------------------


@pytest.fixture
def sqlite_database(
    tmp_path: Path, framework_settings: FrameworkSettings, run_context: RunContext
) -> Iterator[EngineManager]:
    manager = EngineManager(
        sqlite_sandbox_settings(tmp_path / "test.db"),
        environment=run_context.environment,
        redactor=framework_settings.redactor(),
    )
    try:
        yield manager
    finally:
        manager.dispose()  # release the file before pytest removes tmp_path (required on Windows)


@pytest.fixture
def sqlite_memory_database(
    framework_settings: FrameworkSettings, run_context: RunContext
) -> Iterator[EngineManager]:
    manager = EngineManager(
        sqlite_sandbox_settings(MEMORY), environment=run_context.environment, redactor=framework_settings.redactor()
    )
    try:
        yield manager
    finally:
        manager.dispose()


@pytest.fixture
def example_database(sqlite_database: EngineManager) -> EngineManager:
    """A temporary SQLite database holding the example schema and seed data (committed)."""
    with sqlite_database.session() as setup:
        create_example_schema(setup)
        seed_example_data(setup)
        setup.commit()
    return sqlite_database


@pytest.fixture
def example_session(
    example_database: EngineManager, db_session_factory: Callable[..., DatabaseSession]
) -> DatabaseSession:
    return db_session_factory(example_database)


# --- PostgreSQL (explicit opt-in) --------------------------------------------------------------------------------


def _is_postgresql(settings: DatabaseSettings) -> bool:
    try:
        return settings.resolved_dialect == "postgresql"
    except DatabaseError:
        return False


@pytest.fixture
def postgres_database(
    database_settings: DatabaseSettings, framework_settings: FrameworkSettings, run_context: RunContext
) -> Iterator[EngineManager]:
    s = database_settings
    if not (s.is_configured and _is_postgresql(s)):
        pytest.skip("PostgreSQL is not configured (use --cauto-env postgres_test; see docs/database_testing.md)")
    if not (s.access == "read_write" and s.allow_ddl and s.allow_destructive and s.privileged_databases):
        pytest.skip(
            "PostgreSQL tests create and drop a scratch schema: they need access: read_write, allow_ddl, "
            "allow_destructive and the database named in privileged_databases"
        )
    schema = unique_schema_name()
    redactor = framework_settings.redactor()
    bootstrap = EngineManager(s, environment=run_context.environment, redactor=redactor)
    scratch: EngineManager | None = None
    try:
        with bootstrap.session() as admin:  # a configured-but-unreachable server is an error, not a skip
            admin.execute(f"CREATE SCHEMA {schema}")
            admin.commit()
        scratch = EngineManager(
            s.model_copy(update={"schema_name": schema}), environment=run_context.environment, redactor=redactor
        )
        yield scratch
    finally:
        if scratch is not None:
            scratch.dispose()
        try:
            with bootstrap.session() as admin:
                admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                admin.commit()
        finally:
            bootstrap.dispose()


@pytest.fixture
def postgres_session(
    postgres_database: EngineManager, db_session_factory: Callable[..., DatabaseSession]
) -> DatabaseSession:
    """A session on the scratch schema with the example tables created (committed) and seeded."""
    with postgres_database.session() as setup:
        create_example_schema(setup)
        seed_example_data(setup)
        setup.commit()
    return db_session_factory(postgres_database)
