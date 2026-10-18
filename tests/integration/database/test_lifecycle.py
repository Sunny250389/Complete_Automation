"""Engine and session lifecycle: health checks, disposal, pooling, and cleanup that fails."""

from pathlib import Path

import pytest
from sqlalchemy import exc as sa_exc

from complete_automation.database import (
    CleanupError,
    DatabaseSession,
    DatabaseSettings,
    EngineManager,
    UnsafeOperationError,
)
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.validations import count_rows

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]


def test_health_check_reports_the_server_without_credentials(example_database: EngineManager) -> None:
    health = example_database.health_check()

    assert health.ok and health.dialect == "sqlite" and health.server_version and health.latency_ms >= 0
    assert health.error is None and health.database.endswith("test.db")
    assert example_database.open_connections == 0


def test_the_engine_is_created_lazily_and_disposal_is_idempotent(tmp_path: Path) -> None:
    manager = EngineManager(sqlite_sandbox_settings(tmp_path / "lazy.db"))
    assert manager._engine is None  # type: ignore[attr-defined]

    with manager.session() as db:
        db.execute("CREATE TABLE t (a INTEGER)")
        db.commit()
    assert manager._engine is not None  # type: ignore[attr-defined]

    manager.dispose()
    manager.dispose()
    with manager.session() as db:  # a disposed manager can be used again
        assert db.scalar("SELECT count(*) FROM t") == 0
    manager.dispose()


def test_connections_are_returned_to_the_pool_after_every_session(example_database: EngineManager) -> None:
    for _ in range(25):  # more than pool size + overflow would allow if any leaked
        with example_database.session() as db:
            assert count_rows(db, "customers") == 3

    assert example_database.open_connections == 0


def test_concurrent_sessions_use_separate_connections(example_database: EngineManager) -> None:
    first, second = example_database.session(), example_database.session()
    try:
        assert example_database.open_connections == 2 and first.connection is not second.connection
    finally:
        first.close()
        second.close()
    assert example_database.open_connections == 0


def test_closing_twice_is_harmless_and_a_closed_session_cannot_run_statements(example_database: EngineManager) -> None:
    db = example_database.session()
    db.close()
    db.close()

    assert db.closed
    with pytest.raises(Exception):  # noqa: B017 - SQLAlchemy raises ResourceClosedError
        count_rows(db, "customers")


def test_the_connection_context_manager_releases_after_errors(example_database: EngineManager) -> None:
    with pytest.raises(RuntimeError), example_database.connection() as connection:
        connection.exec_driver_sql("SELECT 1")
        raise RuntimeError("test failed")

    assert example_database.open_connections == 0


def test_a_failing_close_is_reported_as_a_cleanup_error(example_database: EngineManager, monkeypatch: pytest.MonkeyPatch) -> None:
    db = example_database.session()

    def broken_close() -> None:
        raise sa_exc.OperationalError("close", {}, Exception("socket reset"))

    monkeypatch.setattr(db.connection, "close", broken_close)

    with pytest.raises(CleanupError, match="connection close failed"):
        db.close()


def test_cleanup_failures_do_not_hide_the_test_failure(example_database: EngineManager, monkeypatch: pytest.MonkeyPatch) -> None:
    db = example_database.session()

    def broken_rollback() -> None:
        raise CleanupError("rollback failed")

    monkeypatch.setattr(db.transactions, "rollback", broken_rollback)
    db.execute("UPDATE customers SET name = 'x' WHERE id = 1")

    with pytest.raises(AssertionError, match="the real failure"), db:
        raise AssertionError("the real failure")  # the cleanup error is logged, not raised over this

    with pytest.raises(CleanupError, match="rollback failed"), example_database.session() as other:
        monkeypatch.setattr(other.transactions, "rollback", broken_rollback)
        other.execute("UPDATE customers SET name = 'y' WHERE id = 1")


def test_a_failed_commit_is_a_transaction_error_and_recorded(example_session: DatabaseSession, monkeypatch: pytest.MonkeyPatch) -> None:
    from complete_automation.database import TransactionError

    def broken_commit() -> None:
        raise sa_exc.OperationalError("commit", {}, Exception("disk full"))

    monkeypatch.setattr(example_session.connection, "commit", broken_commit)

    with pytest.raises(TransactionError, match=r"commit failed \[OperationalError\]"):
        example_session.commit()
    assert example_session.transaction_outcomes[-1].outcome == "failed"


def test_each_manager_validates_its_settings_before_connecting() -> None:
    from complete_automation.database import DatabaseConfigurationError

    with pytest.raises(DatabaseConfigurationError, match="needs modules.database.database"):
        EngineManager(DatabaseSettings(dialect="sqlite"))
    with pytest.raises(DatabaseConfigurationError, match="no adapter for dialect"):
        EngineManager(DatabaseSettings.model_construct(dialect="oracle", url=None, database="x"))  # type: ignore[arg-type]


def test_the_read_only_copy_pattern_gives_an_engine_enforced_read_only_manager(example_database: EngineManager) -> None:
    restricted = example_database.settings.model_copy(
        update={"access": "read_only", "allow_ddl": False, "allow_destructive": False, "privileged_databases": ()}
    )

    with EngineManager(restricted) as ro, ro.session() as db:
        assert count_rows(db, "customers") == 3
        with pytest.raises(UnsafeOperationError):
            db.execute("DELETE FROM customers WHERE id = 1")
