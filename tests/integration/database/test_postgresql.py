"""PostgreSQL integration tests. Opt-in: they skip unless a PostgreSQL test database is configured.

    CAUTO_MODULES__DATABASE__USERNAME=... CAUTO_MODULES__DATABASE__PASSWORD=... \\
        uv run pytest -m postgresql --cauto-env postgres_test

Each test gets a scratch schema (created and dropped by the fixture) holding the example tables, so the
tests never touch other objects in the database. See docs/database_testing.md.
"""

import threading
import time

import pytest

pytest.importorskip("psycopg")

from sqlalchemy import exc as sa_exc  # noqa: E402

from complete_automation.core.exceptions import ErrorCategory  # noqa: E402
from complete_automation.database import (  # noqa: E402
    DatabaseIntegrityError,
    DatabaseSession,
    DataAssertionError,
    EngineManager,
    QueryExecutionError,
    QueryTimeoutError,
    UnsafeOperationError,
    reconcile_queries,
)
from complete_automation.database.example_schema import EXAMPLE_EXPECTATION  # noqa: E402
from complete_automation.database.reconciliation import ReconciliationConfig  # noqa: E402
from complete_automation.database.validations import (  # noqa: E402
    Dependent,
    assert_delete_cascades,
    assert_delete_restricted,
    assert_derived_consistent,
    assert_foreign_keys_enforced,
    assert_row_count,
    assert_schema,
    count_rows,
    snapshot_schema,
)

pytestmark = [
    pytest.mark.database,
    pytest.mark.postgresql,
    pytest.mark.integration,
    pytest.mark.cauto_module("database"),
]

INSERT = "INSERT INTO customers (email, name) VALUES (:e, 'T')"


def _count(manager: EngineManager) -> int:
    with manager.session() as other:  # a separate connection: what committed data looks like
        return count_rows(other, "customers")


# --- connection and identity ----------------------------------------------------------------------------------


def test_connection_health_and_server_identity(postgres_database: EngineManager) -> None:
    health = postgres_database.health_check()

    assert health.ok and health.dialect == "postgresql" and health.server_version
    with postgres_database.session() as db:
        assert postgres_database.verified_database(db.connection) == postgres_database.settings.database
        assert postgres_database.settings.schema_name in db.scalar("SHOW search_path")
    assert postgres_database.open_connections == 0
    assert "password" not in postgres_database.redacted_url.lower() or "***" in postgres_database.redacted_url


def test_foreign_keys_are_always_enforced(postgres_session: DatabaseSession) -> None:
    assert_foreign_keys_enforced(postgres_session)
    assert postgres_session.manager.adapter.capabilities.foreign_keys_always_enforced


# --- CRUD, parameters, dialect SQL -------------------------------------------------------------------------------


def test_crud_with_returning_and_both_parameter_styles(postgres_session: DatabaseSession) -> None:
    new_id = postgres_session.execute("INSERT INTO customers (email, name) VALUES (:e, :n) RETURNING id", {"e": "pg@example.test", "n": "PG"})
    assert new_id.scalar() == 4 and new_id.rowcount == 1

    assert postgres_session.execute("UPDATE customers SET status = ? WHERE id = ?", ["inactive", 4], expect_rowcount=1).rowcount == 1
    assert postgres_session.query("SELECT status FROM customers WHERE id = ?", [4]).scalar() == "inactive"
    assert postgres_session.execute("DELETE FROM customers WHERE id = :i", {"i": 4}, expect_rowcount=1).rowcount == 1
    assert_row_count(postgres_session, "customers", 3)


def test_dialect_specific_sql_passes_through_untranslated(postgres_session: DatabaseSession) -> None:
    assert postgres_session.scalar("SELECT 'abc'::text || 'def'") == "abcdef"
    assert postgres_session.query("SELECT g FROM generate_series(1, 3) AS g").column("g") == [1, 2, 3]
    assert postgres_session.scalar("SELECT count(*) FILTER (WHERE quantity > 1) FROM order_items") >= 0


def test_numeric_columns_are_exact_so_derived_totals_need_no_tolerance(postgres_session: DatabaseSession) -> None:
    assert_derived_consistent(
        postgres_session, parent_table="orders", parent_key="id", parent_column="total",
        child_table="order_items", child_key="order_id", child_column="amount",
    )  # fmt: skip


# --- constraints and error handling ----------------------------------------------------------------------------------


def test_unique_violation_is_classified_without_leaking_the_key_value(postgres_session: DatabaseSession) -> None:
    email = postgres_session.scalar("SELECT email FROM customers WHERE id = 1")

    with pytest.raises(DatabaseIntegrityError) as error:
        postgres_session.execute("INSERT INTO customers (email, name) VALUES (:e, 'Dup')", {"e": email})

    assert error.value.sqlstate == "23505" and error.value.kind == "integrity"
    assert "uq_customers_email" in str(error.value)
    assert email not in str(error.value) and "DETAIL" not in str(error.value)  # PostgreSQL's DETAIL line holds the value
    assert error.value.category is ErrorCategory.ASSERTION and isinstance(error.value.original, sa_exc.IntegrityError)


@pytest.mark.parametrize(
    ("sql", "sqlstate"),
    [
        ("INSERT INTO orders (customer_id, total, placed_at) VALUES (999, 1, now())", "23503"),
        ("INSERT INTO order_items (order_id, sku, quantity, amount) VALUES (1, 'Z', 0, 1)", "23514"),
        ("INSERT INTO customers (email, name) VALUES ('n@example.test', NULL)", "23502"),
    ],
)
def test_foreign_key_check_and_not_null_violations(postgres_session: DatabaseSession, sql: str, sqlstate: str) -> None:
    with pytest.raises(DatabaseIntegrityError) as error:
        postgres_session.execute(sql)

    assert error.value.sqlstate == sqlstate


def test_a_failed_statement_aborts_the_transaction_unless_a_savepoint_contains_it(postgres_session: DatabaseSession) -> None:
    postgres_session.execute(INSERT, {"e": "keep@example.test"})
    with pytest.raises(DatabaseIntegrityError), postgres_session.savepoint():
        postgres_session.execute("INSERT INTO orders (customer_id, total, placed_at) VALUES (999, 1, now())")

    assert_row_count(postgres_session, "customers", 1, {"email": "keep@example.test"})  # the outer transaction survived

    with pytest.raises(DatabaseIntegrityError):  # without a savepoint, the whole transaction is poisoned...
        postgres_session.execute("INSERT INTO orders (customer_id, total, placed_at) VALUES (999, 1, now())")
    with pytest.raises(QueryExecutionError):  # ...and later statements fail until rollback
        postgres_session.scalar("SELECT 1")
    postgres_session.rollback()
    assert postgres_session.scalar("SELECT 1") == 1


def test_referential_actions_behave_as_declared(postgres_session: DatabaseSession) -> None:
    assert_delete_restricted(postgres_session, "customers", {"id": 1})
    assert_delete_cascades(postgres_session, "orders", {"id": 1}, [Dependent("order_items", "order_id", "id")])
    assert count_rows(postgres_session, "orders") == 6


# --- transactions and isolation ------------------------------------------------------------------------------------------


def test_commit_and_rollback(postgres_session: DatabaseSession) -> None:
    manager = postgres_session.manager
    postgres_session.execute(INSERT, {"e": "committed@example.test"})
    assert _count(manager) == 3
    postgres_session.commit()
    assert _count(manager) == 4

    postgres_session.execute(INSERT, {"e": "discarded@example.test"})
    postgres_session.rollback()
    assert _count(manager) == 4


def test_ddl_is_transactional(postgres_database: EngineManager) -> None:
    from complete_automation.database.example_schema import create_example_schema

    assert postgres_database.adapter.capabilities.transactional_ddl
    with postgres_database.session() as db:
        create_example_schema(db)
        assert db.scalar("SELECT count(*) FROM information_schema.tables WHERE table_schema = current_schema()") == 3
        db.rollback()
        assert db.scalar("SELECT count(*) FROM information_schema.tables WHERE table_schema = current_schema()") == 0


def test_repeatable_read_does_not_see_concurrent_commits_but_read_committed_does(postgres_session: DatabaseSession) -> None:
    manager = postgres_session.manager
    with manager.session(isolation_level="REPEATABLE READ") as snapshot, manager.session(isolation_level="READ COMMITTED") as latest:
        assert count_rows(snapshot, "customers") == count_rows(latest, "customers") == 3
        with manager.session(commit_on_success=True) as writer:
            writer.execute(INSERT, {"e": "concurrent@example.test"})

        assert count_rows(snapshot, "customers") == 3  # a stable snapshot
        assert count_rows(latest, "customers") == 4


def test_independent_writers_do_not_block_each_other(postgres_session: DatabaseSession) -> None:
    manager = postgres_session.manager
    assert manager.adapter.capabilities.concurrent_writers
    with manager.session() as one, manager.session() as two:
        one.execute("UPDATE customers SET name = 'one' WHERE id = 1", expect_rowcount=1)
        started = time.monotonic()
        two.execute("UPDATE customers SET name = 'two' WHERE id = 2", expect_rowcount=1)  # SQLite would block here

        assert time.monotonic() - started < 2


def test_deadlocks_are_surfaced_not_retried(postgres_session: DatabaseSession) -> None:
    manager = postgres_session.manager
    barrier = threading.Barrier(2, timeout=15)
    outcomes: dict[str, object] = {}

    def worker(name: str, first: int, second: int) -> None:
        try:
            with manager.session() as db:
                db.execute("UPDATE customers SET name = :n WHERE id = :i", {"n": name, "i": first})
                barrier.wait()
                db.execute("UPDATE customers SET name = :n WHERE id = :i", {"n": name, "i": second})
                db.commit()
                outcomes[name] = "committed"
        except Exception as error:  # noqa: BLE001
            outcomes[name] = error

    threads = [threading.Thread(target=worker, args=("a", 1, 2)), threading.Thread(target=worker, args=("b", 2, 1))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    failures = [o for o in outcomes.values() if isinstance(o, Exception)]
    assert len(outcomes) == 2 and len(failures) == 1 and list(outcomes.values()).count("committed") == 1
    victim = failures[0]
    assert isinstance(victim, QueryExecutionError) and victim.kind == "deadlock" and victim.sqlstate == "40P01"
    assert victim.category is ErrorCategory.INFRASTRUCTURE and victim.original is not None
    assert manager.open_connections == 1  # only the fixture's own session remains


# --- read-only and timeouts ---------------------------------------------------------------------------------------------------


def test_the_server_refuses_writes_on_a_read_only_manager(postgres_session: DatabaseSession) -> None:
    restricted = postgres_session.manager.settings.model_copy(
        update={"access": "read_only", "allow_ddl": False, "allow_destructive": False, "privileged_databases": ()}
    )
    with EngineManager(restricted) as ro, ro.session() as db:
        assert count_rows(db, "customers") == 3
        with pytest.raises(UnsafeOperationError, match="access is read_only"):
            db.execute("DELETE FROM customers WHERE id = 1")
        with pytest.raises(sa_exc.DBAPIError, match="read-only transaction") as error:
            db.connection.exec_driver_sql("DELETE FROM customers WHERE id = 1")  # bypass the policy: the server still refuses
        assert getattr(error.value.orig, "sqlstate", None) == "25006"


def test_statement_timeout_is_enforced_by_the_server(postgres_database: EngineManager) -> None:
    quick = postgres_database.settings.model_copy(update={"statement_timeout_seconds": 0.3})

    with EngineManager(quick) as manager, manager.session() as db:
        started = time.monotonic()
        with pytest.raises(QueryTimeoutError) as error:
            db.query("SELECT pg_sleep(5)")

        assert time.monotonic() - started < 3 and error.value.sqlstate == "57014"
        assert error.value.category is ErrorCategory.TIMEOUT


def test_privileged_statements_require_the_server_reported_database_to_match(postgres_database: EngineManager) -> None:
    wrong = postgres_database.settings.model_copy(update={"privileged_databases": ("some_other_database",)})

    with EngineManager(wrong) as manager, manager.session() as db, pytest.raises(UnsafeOperationError, match="not in privileged_databases"):
        db.execute("CREATE TABLE should_not_exist (a integer)")


# --- schema ----------------------------------------------------------------------------------------------------------------------------


def test_the_example_schema_satisfies_its_expectation_on_postgresql(postgres_session: DatabaseSession) -> None:
    assert_schema(postgres_session, EXAMPLE_EXPECTATION)

    customers = snapshot_schema(postgres_session).table("customers")
    assert customers is not None
    by_name = {c.name: c for c in customers.columns}
    assert by_name["email"].native_type == "VARCHAR(255)" and by_name["email"].max_length == 255
    assert by_name["created_at"].has_default and by_name["status"].default == "active"  # normalized from 'active'::character varying


def test_a_drifted_schema_is_reported_on_postgresql(postgres_session: DatabaseSession) -> None:
    from complete_automation.database import SchemaValidationError

    postgres_session.execute("ALTER TABLE customers ALTER COLUMN name DROP NOT NULL")
    postgres_session.execute("DROP INDEX ix_orders_customer_id")

    with pytest.raises(SchemaValidationError) as error:
        assert_schema(postgres_session, EXAMPLE_EXPECTATION)

    assert "[nullable_mismatch] customers.name" in str(error.value) and "[missing_index] orders(customer_id)" in str(error.value)


# --- across engines ------------------------------------------------------------------------------------------------------------------------


def test_the_same_seed_reconciles_between_sqlite_and_postgresql(example_database: EngineManager, postgres_session: DatabaseSession) -> None:
    config = ReconciliationConfig(key=("id",), default_numeric_tolerance=0.005, ignore_fields=frozenset({"placed_at"}))
    query = "SELECT id, customer_id, total, status FROM orders ORDER BY id"

    with example_database.session() as sqlite_db:
        result = reconcile_queries(sqlite_db, query, postgres_session, query, config)

    result.assert_matches()
    assert result.matched == 6


def test_reconciliation_fails_when_the_engines_disagree(example_database: EngineManager, postgres_session: DatabaseSession) -> None:
    postgres_session.execute("UPDATE orders SET status = 'cancelled' WHERE id = 3")
    config = ReconciliationConfig(key=("id",), default_numeric_tolerance=0.005)
    query = "SELECT id, status FROM orders ORDER BY id"

    with example_database.session() as sqlite_db, pytest.raises(DataAssertionError, match=r"mismatched 1"):
        reconcile_queries(sqlite_db, query, postgres_session, query, config).assert_matches()

