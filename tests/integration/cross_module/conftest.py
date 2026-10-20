"""Cross-module workflows: the API, database and UI modules against one local application.

Needs the "api" and "database" extras; the browser workflow also needs the "ui" extra
and an installed browser (its tests are marked ``ui``).

The API and database plugins are *registered* here rather than star-imported: both
define a ``pytest_runtest_makereport`` hook, and in one module namespace the second
would replace the first, silently switching off the API module's failure capture.
Registered as plugins, each keeps its own hook.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

try:
    import httpx, sqlalchemy  # noqa: E401,F401
except ImportError:  # the "api" or "database" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.api import pytest_plugin as api_plugin
    from complete_automation.core.config import FrameworkSettings
    from complete_automation.core.context import RunContext
    from complete_automation.database import pytest_plugin as database_plugin
    from complete_automation.database.engine import EngineManager
    from complete_automation.database.session import DatabaseSession
    from complete_automation.database.settings import DatabaseSettings
    from tests.support.users_app import FIXED_NOW, SEED_USERS, UsersApp

    def pytest_configure(config: pytest.Config) -> None:
        for plugin in (api_plugin, database_plugin):
            if not config.pluginmanager.is_registered(plugin):
                config.pluginmanager.register(plugin, plugin.__name__)

    @pytest.fixture
    def users_app(tmp_path: Path) -> Iterator[UsersApp]:
        """The application on 127.0.0.1 with its own database file, seeded and removed per test."""
        app = UsersApp(tmp_path / "users_app.db", now=lambda: FIXED_NOW)
        for name, email, role in SEED_USERS:
            app.store.create(name, email, role)
        try:
            yield app
        finally:
            app.stop()

    @pytest.fixture
    def users_db(
        users_app: UsersApp, framework_settings: FrameworkSettings, run_context: RunContext
    ) -> Iterator[EngineManager]:
        """The application's database through the database module, opened read-only.

        The application is the only writer; the test can only observe. SQLite opens the
        file with ``mode=ro``, so even a faulty test cannot change the data it verifies.
        """
        settings = DatabaseSettings(dialect="sqlite", database=str(users_app.db_path), access="read_only")
        manager = EngineManager(settings, environment=run_context.environment, redactor=framework_settings.redactor())
        try:
            yield manager
        finally:
            manager.dispose()  # release the file before pytest removes tmp_path (required on Windows)

    @pytest.fixture
    def users_db_session(
        users_db: EngineManager, db_session_factory: Callable[..., DatabaseSession]
    ) -> DatabaseSession:
        """A read-only session whose statements are recorded in the test's result."""
        return db_session_factory(users_db)
