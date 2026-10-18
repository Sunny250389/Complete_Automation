"""Commit, rollback, savepoints, failure cleanup and isolation, on real SQLite databases.

SQLite has a single writer and serialized transactions, so these tests verify the
framework's transaction contract on SQLite. They say nothing about PostgreSQL
concurrency; see test_postgresql.py for engine-specific behavior.
"""

import pytest

from complete_automation.database import (
    DatabaseConfigurationError,
    DatabaseSession,
    EngineManager,
    TransactionError,
)
from complete_automation.database.example_schema import create_example_schema
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.validations import assert_row_absent, assert_row_count, count_rows

pytestmark = [pytest.mark.database, pytest.mark.integration, pytest.mark.cauto_module("database")]

INSERT = "INSERT INTO customers (email, name) VALUES (:e, 'T')"


def _count(manager: EngineManager) -> int:
    """Count customers through a *separate* connection, i.e. what committed data looks like."""
    with manager.session() as other:
        return count_rows(other, "customers")


def test_commit_persists_changes_for_other_connections(example_database: EngineManager) -> None:
    with example_database.session() as db:
        db.execute(INSERT, {"e": "committed@example.test"})
        assert _count(example_database) == 3  # not visible before the commit
        db.commit()

    assert _count(example_database) == 4


def test_explicit_rollback_discards_changes(example_database: EngineManager) -> None:
    with example_database.session() as db:
        db.execute(INSERT, {"e": "discarded@example.test"})
        assert_row_count(db, "customers", 4)  # visible to its own transaction
        db.rollback()

        assert_row_count(db, "customers", 3)

    assert _count(example_database) == 3
    assert [o.outcome for o in db.transaction_outcomes] == ["rolled_back", "rolled_back"]  # explicit, then at close


def test_leaving_the_session_rolls_back_uncommitted_work(example_database: EngineManager) -> None:
    with example_database.session() as db:
        db.execute(INSERT, {"e": "forgotten@example.test"})

    assert _count(example_database) == 3
    assert db.closed and example_database.open_connections == 0


def test_nothing_is_committed_automatically(example_database: EngineManager) -> None:
    db = example_database.session()
    db.execute(INSERT, {"e": "implicit@example.test"})

    assert _count(example_database) == 3
    db.close()
    assert _count(example_database) == 3


def test_commit_on_success_is_explicit_and_skipped_on_errors(example_database: EngineManager) -> None:
    with example_database.session(commit_on_success=True) as db:
        db.execute(INSERT, {"e": "ok@example.test"})
    with pytest.raises(RuntimeError), example_database.session(commit_on_success=True) as db:
        db.execute(INSERT, {"e": "failed@example.test"})
        raise RuntimeError("test body failed")

    assert _count(example_database) == 4
    with example_database.session() as check:
        assert_row_absent(check, "customers", {"email": "failed@example.test"})


def test_an_exception_in_the_body_rolls_back_and_releases_the_connection(example_database: EngineManager) -> None:
    with pytest.raises(AssertionError, match="boom"), example_database.session() as db:
        db.execute(INSERT, {"e": "partial@example.test"})
        raise AssertionError("boom")

    assert db.closed and example_database.open_connections == 0 and _count(example_database) == 3


def test_a_failed_statement_does_not_leave_a_half_applied_transaction(example_database: EngineManager) -> None:
    with example_database.session() as db:
        db.execute(INSERT, {"e": "first@example.test"})
        with pytest.raises(Exception, match="UNIQUE"):
            db.execute(INSERT, {"e": "first@example.test"})
        db.rollback()

    assert _count(example_database) == 3


def test_transaction_block_rolls_back_by_default_and_commits_on_request(example_database: EngineManager) -> None:
    with example_database.session() as db:
        with db.transaction():
            db.execute(INSERT, {"e": "peek@example.test"})
            assert_row_count(db, "customers", 4)
        db.rollback()  # the check above started a new implicit transaction
        assert_row_count(db, "customers", 3)
        db.rollback()

        with db.transaction(commit=True):
            db.execute(INSERT, {"e": "keep@example.test"})

    assert _count(example_database) == 4


def test_transaction_block_rolls_back_on_error_and_reraises(example_database: EngineManager) -> None:
    with example_database.session() as db:
        with pytest.raises(ValueError, match="bad"), db.transaction(commit=True):
            db.execute(INSERT, {"e": "never@example.test"})
            raise ValueError("bad")

        assert not db.transactions.active
        assert_row_count(db, "customers", 3)


def test_a_transaction_block_cannot_adopt_statements_that_already_ran(example_database: EngineManager) -> None:
    with example_database.session() as db:
        count_rows(db, "customers")  # starts an implicit transaction

        with pytest.raises(TransactionError, match="already active"), db.transaction():
            pass  # pragma: no cover - never entered


def test_beginning_twice_is_a_transaction_error(example_database: EngineManager) -> None:
    with example_database.session() as db:
        db.transactions.begin()
        with pytest.raises(TransactionError, match="already active"):
            db.transactions.begin()


# --- savepoints --------------------------------------------------------------------------------------


def test_savepoint_rolls_back_only_the_nested_work_on_error(example_session: DatabaseSession) -> None:
    example_session.execute(INSERT, {"e": "outer@example.test"})

    with pytest.raises(Exception, match="UNIQUE"), example_session.savepoint():
        example_session.execute(INSERT, {"e": "inner@example.test"})
        example_session.execute(INSERT, {"e": "inner@example.test"})  # violates uniqueness

    assert_row_count(example_session, "customers", 1, {"email": "outer@example.test"})
    assert_row_absent(example_session, "customers", {"email": "inner@example.test"})
    assert example_session.transactions.active  # the outer transaction survived


def test_savepoint_releases_on_success_and_inspect_then_undo_discards(example_session: DatabaseSession) -> None:
    with example_session.savepoint():
        example_session.execute(INSERT, {"e": "kept@example.test"})
    with example_session.rolled_back_savepoint():
        example_session.execute(INSERT, {"e": "peeked@example.test"})
        assert_row_count(example_session, "customers", 5)

    assert_row_count(example_session, "customers", 4)
    assert [(o.scope, o.outcome) for o in example_session.transaction_outcomes] == [
        ("savepoint", "committed"),
        ("savepoint", "rolled_back"),
    ]


def test_savepoints_nest(example_session: DatabaseSession) -> None:
    with example_session.savepoint():
        example_session.execute(INSERT, {"e": "level1@example.test"})
        with pytest.raises(ZeroDivisionError), example_session.savepoint():
            example_session.execute(INSERT, {"e": "level2@example.test"})
            1 / 0  # noqa: B018

    assert_row_count(example_session, "customers", 1, {"email": "level1@example.test"})
    assert_row_absent(example_session, "customers", {"email": "level2@example.test"})


# --- DDL, isolation, independence -------------------------------------------------------------------------


def test_sqlite_ddl_is_transactional(sqlite_database: EngineManager) -> None:
    assert sqlite_database.adapter.capabilities.transactional_ddl
    with sqlite_database.session() as db:
        create_example_schema(db)
        db.rollback()

        assert db.query("SELECT name FROM sqlite_master WHERE type = 'table'").rows == ()


def test_isolation_level_is_validated_per_engine(example_database: EngineManager) -> None:
    with example_database.session(isolation_level="SERIALIZABLE") as db:
        assert count_rows(db, "customers") == 3
    with pytest.raises(DatabaseConfigurationError, match="REPEATABLE READ"):
        example_database.session(isolation_level="REPEATABLE READ")
    with pytest.raises(DatabaseConfigurationError, match="AUTOCOMMIT"):
        example_database.session(isolation_level="AUTOCOMMIT")
    assert example_database.open_connections == 0  # a refused session leaks nothing


def test_tests_do_not_contaminate_each_other_first(example_database: EngineManager) -> None:
    """Paired with ..._second: each test gets its own database, whichever runs first."""
    with example_database.session(commit_on_success=True) as db:
        db.execute(INSERT, {"e": "shared-name@example.test"})

    assert _count(example_database) == 4


def test_tests_do_not_contaminate_each_other_second(example_database: EngineManager) -> None:
    with example_database.session() as db:
        assert_row_absent(db, "customers", {"email": "shared-name@example.test"})
        assert_row_count(db, "customers", 3)


def test_memory_databases_share_one_connection_so_use_files_to_test_isolation(sqlite_memory_database: EngineManager) -> None:
    assert sqlite_memory_database.adapter.in_memory  # type: ignore[attr-defined]
    with sqlite_memory_database.session() as first:
        first.execute("CREATE TABLE m (v INTEGER)")
        first.execute("INSERT INTO m VALUES (1)")
        first.commit()
    with sqlite_memory_database.session() as second:
        assert second.scalar("SELECT count(*) FROM m") == 1  # the same database, not a fresh empty one


def test_a_file_database_keeps_data_across_engines(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = sqlite_sandbox_settings(tmp_path / "persist.db")
    with EngineManager(settings) as first, first.session() as db:
        db.execute("CREATE TABLE p (v INTEGER)")
        db.execute("INSERT INTO p VALUES (7)")
        db.commit()

    with EngineManager(settings) as second, second.session() as db:
        assert db.scalar("SELECT v FROM p") == 7
