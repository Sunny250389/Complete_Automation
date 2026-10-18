# Database testing

`complete_automation.database` tests relational databases: connection and transaction handling, safe SQL execution, CRUD verification, data and referential integrity checks, schema and migration validation, and record reconciliation. SQLite is fully supported and needs no server. PostgreSQL is supported through the same adapter contract.

Everything shared comes from [the shared core](shared_core.md): configuration, logging and redaction, results, reporting and artifacts. The database module adds no second version of any of them.

**Boundaries.**

- The module depends only on `complete_automation.core`, SQLAlchemy and (for PostgreSQL) psycopg.
- It contains no HTTP, browser, ML or LLM code. The core and the API module never import it.
- SQL execution lives only in `complete_automation/database/`, never in `core/` or `api/`.
- `tests/unit/database/test_architecture.py` enforces these rules and checks that importing the package, or constructing an `EngineManager`, opens no connection.

## Installation

```bash
uv sync --locked --extra database      # SQLAlchemy; enough for SQLite
uv sync --locked --extra postgresql    # SQLAlchemy + psycopg 3 for PostgreSQL
uv sync --locked --all-extras          # everything, as CI does
```

Without the extra, the core still works and the database test folders are left out of collection. The PostgreSQL driver is imported only when a PostgreSQL engine is created, so a missing driver is reported as a configuration error with the install command, not as an import failure.

## Architecture

| Module | Public interface |
| --- | --- |
| `database/settings.py` | `DatabaseSettings` (the `modules.database` section), `safe_url` |
| `database/adapters/` | `DatabaseAdapter`, `Capabilities`, `SqliteAdapter`, `PostgreSqlAdapter`, `register_adapter`, `get_adapter_class` |
| `database/engine.py` | `EngineManager`, `HealthReport` |
| `database/session.py` | `DatabaseSession` (one connection, an executor and explicit transactions) |
| `database/executor.py` | `SqlExecutor`, `QueryResult` |
| `database/transactions.py` | `TransactionManager`, `TransactionOutcome` |
| `database/safety.py`, `database/sqltext.py` | `OperationPolicy`, `OperationKind`, `classify`, `fingerprint` |
| `database/errors.py`, `database/classification.py` | Error hierarchy, `wrap_error` |
| `database/validations/assertions.py` | Data assertions (counts, records, nulls, duplicates, ranges, aggregates, pagination) |
| `database/validations/schema.py` | `SchemaExpectation` models, `snapshot_schema`, `validate_schema`, `assert_schema` |
| `database/validations/integrity.py` | Orphans, cascade and restrict checks, derived fields, source/target aggregates |
| `database/validations/migrations.py` | Snapshot diffs, row-count preservation, schema version |
| `database/validations/identifiers.py` | `safe_identifier` and the expression builders behind the assertions |
| `database/reconciliation.py` | `reconcile`, `reconcile_ordered`, `reconcile_queries`, `ReconciliationConfig` |
| `database/reporting.py` | `record_query`, `record_transactions`, `record_connection`, `save_diagnostics` |
| `database/example_schema.py` | The example schema, its `EXAMPLE_EXPECTATION`, deterministic seed data |
| `database/sandbox.py` | Settings for databases a test creates and owns |
| `database/pytest_plugin.py` | Fixtures (see [Fixtures](#fixtures-and-scopes)) |

The layers, bottom to top: **adapter** (what differs per engine) → **`EngineManager`** (engine lifecycle) → **`DatabaseSession`** (a connection with the executor and transaction manager) → **validations** and **reconciliation** (engine-independent, written against a session).

## Configuration

Settings live under `modules.database`. Values come from `config/environments/<env>.yaml`, then `CAUTO_*` values in `.env`, then `CAUTO_*` environment variables, as for every module. **Secrets belong in environment variables or `.env`, never in YAML.**

```yaml
modules:
  database:
    dialect: postgresql
    host: db.example.test
    port: 5432
    database: app_test
    username: app_tester          # password: CAUTO_MODULES__DATABASE__PASSWORD
    schema: app                   # search_path (PostgreSQL)
    access: read_only             # the default
```

```bash
CAUTO_MODULES__DATABASE__PASSWORD=...      # a SecretStr: never shown in repr, logs or reports
CAUTO_MODULES__DATABASE__URL=postgresql+psycopg://user:pw@host/db   # alternative to the fields above
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `dialect` | from `url` | `sqlite` or `postgresql` |
| `url` | none | A full SQLAlchemy URL (secret). Cannot be combined with `host`, `port`, `database`, `username`, `password` |
| `database` | none | PostgreSQL database name, or the SQLite file (`:memory:` for in-memory). The fixtures (and `settings.with_base_dir(...)`) resolve relative file paths against the project root |
| `host`, `port`, `username`, `password` | none | PostgreSQL connection fields |
| `schema` | none | PostgreSQL schema put first on the `search_path` |
| `pool.size`, `pool.max_overflow` | 5, 10 | Pooled connections (file SQLite and PostgreSQL) |
| `pool.timeout_seconds` | 30 | How long to wait for a free pooled connection |
| `pool.recycle_seconds`, `pool.pre_ping` | 1800, true | Replace stale connections |
| `connect_timeout_seconds` | 10 | Connection timeout (PostgreSQL); busy timeout (SQLite) |
| `statement_timeout_seconds` | none | Server-side statement timeout. PostgreSQL only; SQLite rejects it |
| `isolation_level` | engine default | One of the adapter's levels. `AUTOCOMMIT` is never allowed |
| `tls.mode`, `tls.ca_bundle` | none | PostgreSQL `sslmode` and root certificate |
| `sqlite.foreign_keys` | true | Enforce foreign keys (SQLite ignores them otherwise) |
| `sqlite.journal_mode` | none | e.g. `wal` (file databases opened read-write) |
| `access` | `read_only` | `read_only` or `read_write` |
| `allow_ddl`, `allow_destructive` | false | See [Read-only mode and destructive safeguards](#read-only-mode-and-destructive-safeguards) |
| `privileged_databases` | `[]` | Databases on which DDL and destructive statements may run |
| `protected_environments` | `[prod, production]` | Environments that are always read-only |
| `max_rows` | 100000 | Rows a non-streaming query may return |
| `capture_diagnostics` | `on_failure` | `never`, `on_failure` or `always` |

Invalid or contradictory settings fail at configuration time with a message that names the setting and never echoes a value. A bad `url` is reported without printing it.

## Supported engines and limitations

| | SQLite | PostgreSQL |
| --- | --- | --- |
| Driver | `pysqlite` (standard library) | `psycopg` 3 (`postgresql` extra) |
| Needs a server | no | yes |
| Savepoints | yes | yes |
| Transactional DDL | yes | yes |
| Concurrent writers | **no** (one writer) | yes |
| Foreign keys | only with `PRAGMA foreign_keys` (enabled by default here) | always |
| Statement timeout | no | yes (`statement_timeout_seconds`) |
| Schemas | no | yes (`schema`) |
| Isolation levels | `SERIALIZABLE`, `READ UNCOMMITTED` | `READ UNCOMMITTED` (acts as `READ COMMITTED`), `READ COMMITTED`, `REPEATABLE READ`, `SERIALIZABLE` |
| Engine-enforced read-only | `mode=ro` for files, `PRAGMA query_only` always | `default_transaction_read_only` on every connection |
| Column types | dynamic typing (affinities); `NUMERIC` is floating point | strict; `NUMERIC` is exact |

Read the capabilities in code with `manager.adapter.capabilities`. Do not read SQLite behavior as evidence of PostgreSQL behavior: concurrency, isolation and locking tests belong against PostgreSQL.

Adapters never translate SQL. A statement is sent as written, so dialect-specific SQL (`::text`, `generate_series`, `FILTER`) stays visible in the test that uses it. Unsupported options (a `schema` or `statement_timeout_seconds` on SQLite) fail with a `DatabaseConfigurationError`, and unsupported operations raise `UnsupportedOperationError`.

**Adding an engine.** Subclass `DatabaseAdapter` (URL and engine options, per-connection setup, server identity, error classification, capabilities) and call `register_adapter`. The executor, transactions, validations and reconciliation work unchanged, because they use only SQLAlchemy Core connections. MySQL, SQL Server and Oracle are not implemented.

## Connection lifecycle

```python
from complete_automation.database import DatabaseSettings, EngineManager

settings = DatabaseSettings(dialect="sqlite", database="data/app.db")     # nothing opened yet
with EngineManager(settings, environment="dev") as manager:               # still nothing opened
    with manager.session() as db:                                          # connects here
        db.query("SELECT 1")
# leaving the block rolled back anything uncommitted, returned the connection and disposed the engine
```

- Constructing an `EngineManager` validates settings and builds the URL. The SQLAlchemy engine is created on first use, and the first connection opens when a session is requested. Importing the package connects to nothing.
- **Pooling:** file SQLite and PostgreSQL use SQLAlchemy's `QueuePool`. **In-memory SQLite uses one shared connection** (`StaticPool`), because every new connection to `:memory:` would otherwise be a separate, empty database. All sessions then share transaction state; use a file database to test commits and isolation across connections.
- `manager.open_connections` counts checked-out connections. Zero after a test means nothing leaked.
- `manager.health_check()` returns a `HealthReport` (server version, latency, error kind) and does not raise for connection problems. It never contains credentials.
- `manager.dispose()` closes every pooled connection. It is idempotent, and the manager can be used again afterwards. On Windows, dispose a SQLite engine before deleting its file.
- A `DatabaseSession` wraps one connection and is **not thread-safe**. Use one session per thread or test.

## Executing SQL

```python
db.query("SELECT id, email FROM customers WHERE id >= :low ORDER BY id", {"low": 2})   # named
db.query("SELECT id FROM customers WHERE id = ? OR id = ?", [1, 3])                    # positional
db.execute("INSERT INTO customers (email, name) VALUES (:e, :n)", {"e": e, "n": n}, expect_rowcount=1)
db.execute_many(insert(customers), [{"email": "a@x.test", "name": "A"}, ...])        # batch
for row in db.iterate("SELECT * FROM big_table ORDER BY id", chunk_size=1000): ...    # streaming
db.scalar("SELECT count(*) FROM customers")
```

- `query` returns a `QueryResult`: `columns`, `rows` (dicts), `row_count`, `rowcount`, `duration_ms`, `operation`, plus `first()`, `one()`, `scalar()`, `column(name)`. `fetch_all`, `fetch_one` and `scalar` are shortcuts.
- SQLAlchemy Core statements (`select()`, `insert()`, `text()`) work everywhere a string does. Raw SQL stays available for engine-specific features.
- **Values are always bound parameters.** Positional `?` placeholders become named binds outside literals and comments. Two cautions: a literal colon in SQL text needs escaping as `\:` (SQLAlchemy `text()` rule), and PostgreSQL's JSONB `?` operator conflicts with positional placeholders, so use named parameters in such statements.
- **Identifiers are not parameters.** The validation helpers check table, column and schema names with `safe_identifier` (plain `[A-Za-z0-9_$]` names, optional allowlist) and then build SQLAlchemy expressions, which quote correctly. Do the same in your own code: never format a name into SQL text.
- `query` and `iterate` accept only reads. `execute` handles writes and, when authorized, DDL. Nothing commits automatically, and writes are never retried.
- `expect_rowcount` raises `DataAssertionError` when the number of affected rows differs. The statement has already run inside the open transaction, so roll back.
- A result larger than `max_rows` raises `ResultTooLargeError`; stream it with `iterate` instead of raising the limit blindly.
- `RETURNING` works on both engines. The affected-row count of such a statement is the number of returned rows.

### Errors and diagnostics

Driver errors are wrapped in classified `DatabaseError` subclasses. A message keeps the first line of the driver message, the exception type and the SQLSTATE, and drops the SQL text, bound values and PostgreSQL's `DETAIL` line (which contains the offending key value). The original exception is on `.original`. It is deliberately not chained with `raise ... from`, because pytest would print the chained driver message into the report.

| Error | Category | Typical cause |
| --- | --- | --- |
| `DataAssertionError`, `SchemaValidationError` | assertion | A data or schema check failed |
| `DatabaseIntegrityError` | assertion | A constraint rejected the statement |
| `UnsafeOperationError`, `DatabaseConfigurationError`, `UnsupportedOperationError` | configuration | Policy refused a statement, bad settings |
| `DatabaseConnectionError`, `CleanupError`, `TransactionError` | infrastructure | Connection lost, rollback or close failed |
| `QueryTimeoutError` | timeout | Statement, lock or pool wait exceeded its limit |
| `QueryExecutionError` | by `kind` | `syntax`, `permission`, `data`, `deadlock`, `serialization`, `other` |
| `ResultTooLargeError`, `ReconciliationError` | test data | Unsuitable input |

Deadlocks and serialization failures surface as `QueryExecutionError` with `kind="deadlock"` / `"serialization"`. Nothing in the module retries a transaction.

## CRUD verification

All helpers are in `complete_automation.database.validations`. They take a session, a table name and optional `schema`, and fail with a message naming the check, the table, and the expected and actual values.

| | Helpers |
| --- | --- |
| Create | `assert_row_exists` (returns the row), `assert_record` (field values, optional `tolerance`), generated ids via `RETURNING`, defaults and constraints through the usual assertions |
| Read | `assert_row_exists`, `count_rows`, `assert_row_count`, `assert_row_count_between`, `assert_table_empty`, `assert_table_not_empty`, `assert_sorted`, `assert_pagination_consistent`, `assert_aggregate` |
| Update | `execute(..., expect_rowcount=n)`, then `assert_record` |
| Delete | `execute(..., expect_rowcount=n)`, `assert_row_absent`, `assert_delete_restricted`, `assert_delete_cascades` |

Conditions are mappings: `{"status": "open"}` is `status = :v`, `{"deleted_at": None}` is `IS NULL`, `{"id": [1, 2]}` is `IN`.

Further record and collection checks: `assert_not_null`, `assert_required_fields`, `find_duplicates`, `assert_no_duplicates`, `assert_values_in_range` (numbers, dates, timestamps), `assert_string_length`, `assert_values_in_set`, `assert_distinct_count`, `assert_grouped_counts`, `assert_recent` (takes an explicit `now`).

**Failure messages stay safe.** Only compared columns appear (never whole rows), values of sensitive column names (`password`, `token`, ...) are masked by the shared redactor, long values are truncated, and duplicate values are hidden unless `show_values=True`.

## Integrity checks

- `assert_foreign_keys_enforced(db)`: fail fast if the connection does not enforce foreign keys, so constraint tests cannot pass for the wrong reason.
- `find_orphans` / `assert_no_orphans`: child rows whose non-NULL foreign key has no parent. NULL foreign keys are not orphans.
- `assert_delete_restricted`, `assert_delete_cascades`: delete inside a savepoint that is always rolled back, then check that the engine restricted the delete or removed exactly the dependent rows. They need write access and savepoint support.
- `find_derived_mismatches` / `assert_derived_consistent`: a stored value against `sum`/`count`/`min`/`max` over child rows (for example `orders.total` against `sum(order_items.amount)`).
- `assert_aggregate_matches`: the same aggregate over a source and a target table.

SQLite stores `NUMERIC` as floating point, so derived sums can differ in the last digit (`242.84` against `242.84000000000003`). Pass `tolerance=0.005`. PostgreSQL's `NUMERIC` is exact and needs no tolerance.

## Schema validation

```python
from complete_automation.database.validations import (
    ColumnExpectation, ForeignKeyExpectation, SchemaExpectation, TableExpectation, assert_schema)

expected = SchemaExpectation(tables=(
    TableExpectation(
        name="authors",
        columns=(ColumnExpectation(name="email", type="string", nullable=False, max_length=100),),
        primary_key=("id",),
        unique_constraints=(("email",),),
    ),
    TableExpectation(name="books", foreign_keys=(
        ForeignKeyExpectation(columns=("author_id",), referred_table="authors", referred_columns=("id",)),)),
))
assert_schema(db, expected)   # SchemaValidationError lists every difference
```

- Anything an expectation leaves unset is not checked. Extra tables and columns are tolerated unless `exact_tables` / `exact_columns` are set. Two schemas are never treated as equal because their table names match.
- **Types are compared as families** (`integer`, `numeric`, `float`, `string`, `boolean`, `date`, `datetime`, `time`, `binary`, `json`, `uuid`) so SQLite `VARCHAR(10)` and PostgreSQL `character varying(10)` both count as `string`. Add `native_type` to also require the engine's reflected type name.
- Checked: tables, views, columns, family, native type, nullability, primary key, length, default presence and value (quotes and `::casts` normalized), foreign keys (and `on_delete`), unique constraints, indexes (columns, uniqueness, name).
- Expectations are pydantic models, so they load from YAML: `SchemaExpectation.model_validate(load_yaml(path))`.
- `snapshot_schema(db)` reflects the whole schema into a fully populated expectation. Validation only reads metadata and never alters the database.
- Not compared: `CHECK` constraints, expression indexes (skipped), triggers, sequences, privileges. Engine-specific types such as enums appear in the `string` or `other` family.

### Migration checks

The module **validates** migrations; it never runs them. Apply them with your migration tool against a database you control.

```python
before, counts = snapshot_schema(db), table_counts(db, ["customers", "orders"])
run_your_migration()
assert_schema_changes(before, snapshot_schema(db), added_tables=["audit_log"], added_columns={"customers": ["phone"]})
assert_counts_preserved(counts, table_counts(db, ["customers", "orders"]))
assert_schema_version(db, "1a2b3c")        # Alembic's alembic_version.version_num by default
```

`assert_schema_changes` requires the change set to equal the declaration: an undeclared change fails as well as a missing one. No migration tool is installed in this repository, so there is no Alembic integration beyond reading the version table.

## Transactions

```python
db.execute(...)
db.commit()                      # nothing commits unless this is called
db.rollback()

with db.savepoint():             # nested transaction: rolled back on error, released on success
    ...
with db.rolled_back_savepoint(): # inspect, then undo
    ...
with db.transaction(commit=True):  # all-or-nothing block; commit=False (the default) rolls back
    ...
```

- Leaving a session rolls back uncommitted work and releases the connection, including after assertion failures. `session(commit_on_success=True)` commits only if the block completes without an exception.
- Every statement starts a transaction implicitly (SQLAlchemy "autobegin"). `transaction()` therefore needs a connection with nothing in progress; call `commit()` or `rollback()` first, or it raises `TransactionError`.
- A failed rollback raises `CleanupError`, but never over a test failure that is already propagating (the cleanup problem is logged instead). `session.transaction_outcomes` lists commits, rollbacks and failures.
- `session(isolation_level=...)` validates the level against the adapter and rejects `AUTOCOMMIT`.
- **DDL** is transactional on both supported engines (SQLite and PostgreSQL), so a rolled-back migration step is undone. Other engines differ (MySQL commits implicitly on DDL), which is why `adapter.capabilities.transactional_ddl` exists.
- **After an error in PostgreSQL the transaction is aborted** and later statements fail until rollback. Wrap statements that are expected to fail in `db.savepoint()`. SQLite does not poison the transaction.
- SQLite keeps one writer and serializes transactions. Tests of concurrent writes, deadlocks and MVCC isolation need PostgreSQL (`test_postgresql.py` covers repeatable read vs read committed and a real deadlock).

## Read-only mode and destructive safeguards

Three layers, strongest first:

1. **The database account.** Use least-privilege credentials. Nothing in this module replaces that.
2. **Engine-level read-only**, applied by the adapter to every connection when `access: read_only`: SQLite opens files with `mode=ro` (which also refuses to create a missing file) and sets `PRAGMA query_only`; PostgreSQL sets `default_transaction_read_only`. The server refuses writes even when the application check is bypassed, including side-effecting reads such as `nextval()`/`setval()`. Advisory locks and other session state are not blocked.
3. **The operation policy**, which classifies each statement (`READ`, `WRITE`, `DDL`, `DESTRUCTIVE`, `ADMIN`, `CONTROL`) and refuses what the settings do not authorize, with a message that says why. Classification is keyword level: it understands literals, comments, quoted identifiers and `$$` quoting, so keywords inside them never matter, but it is a guardrail against mistakes. It is not a defense against hostile SQL.

| Operation | Needs |
| --- | --- |
| `SELECT`, `WITH ... SELECT`, `EXPLAIN` | nothing |
| `INSERT`, `UPDATE`, `DELETE ... WHERE` | `access: read_write` |
| `CREATE`, `ALTER`, unrecognized or privileged statements (`GRANT`, `COPY`, `PRAGMA x = y`) | also `allow_ddl` and the database in `privileged_databases` |
| `DROP`, `TRUNCATE`, `DELETE` without `WHERE`, `ALTER ... DROP` | also `allow_destructive` and the database in `privileged_databases` |
| `COMMIT`, `ROLLBACK`, `BEGIN`, `SAVEPOINT`, `SET ...` | never as raw SQL; use the transaction API |
| Several statements in one call | never |

- Before DDL or a destructive statement, the executor asks the **server** which database it is connected to and requires it to be in `privileged_databases` and to equal the configured database. A mistyped host cannot redirect a `DROP` at another database.
- **Protected environments** (`prod`, `production` by default) are always read-only. Configuring `access: read_write`, `allow_ddl` or `allow_destructive` there raises `DatabaseConfigurationError` when the manager is created.
- `privileged_databases` entries are compared as whole names (or file names for SQLite), never as substrings.
- Failed or refused statements never include credentials. Connection URLs are redacted wherever they surface (`manager.redacted_url`, `repr`, health checks, errors).

## Test data and isolation

| Strategy | Use when | Trade-off |
| --- | --- | --- |
| Temporary SQLite file (`sqlite_database`, `example_database`) | Most tests; anything that needs commits, several connections or schema changes | A fresh database per test: fully isolated, slightly more setup |
| In-memory SQLite (`sqlite_memory_database`) | Fast single-connection tests | Sessions share one connection, so no cross-connection isolation |
| Transaction rollback (`db_session` on a configured database) | Tests against a shared database that only need to see their own writes | Cannot test persistence or concurrency; a test that commits leaves data behind |
| Scratch PostgreSQL schema (`postgres_database`) | PostgreSQL tests that create and drop objects | Needs DDL rights on a disposable database |
| Unique test records | Tests against a shared database that must commit | You must clean up explicitly |

- Reuse the shared core's `DataFactory` and `DataLoader` for data; `seed_example_data` shows deterministic seeding with `DataFactory`. The module adds no data framework.
- Do not assume rollback works for every test: commits, other connections and concurrency need committed data.
- Never run a broad `DELETE` or `TRUNCATE` on a shared database. The policy refuses bulk deletes unless a privileged, allowlisted database is configured.
- Database sandboxes (`sqlite_sandbox_settings`) are for databases the test creates and owns. Never build them for a shared or pre-existing database.

## Reconciliation

```python
from complete_automation.database import ReconciliationConfig, reconcile, reconcile_queries

config = ReconciliationConfig(key=("id",), ignore_fields=frozenset({"updated_at"}), default_numeric_tolerance=0.005)
result = reconcile(expected_rows, db.query("SELECT id, name FROM customers").rows, config)
result.assert_matches()

# two databases, constant memory: both queries must ORDER BY the key, ascending
result = reconcile_queries(source_db, "SELECT ... ORDER BY id", target_db, "SELECT ... ORDER BY id", config)
```

- The key is explicit and may be composite. Missing, unexpected, mismatched and duplicate-key records are counted and reported **separately**; nothing is silently ignored. Samples are bounded by `max_samples`, counts are complete.
- Explicit rules: `ignore_fields`, `compare_fields`, per-field `tolerances` (numbers; seconds for datetimes), `default_numeric_tolerance`, `normalizers` per field, and `naive_datetimes_as_utc` (off by default, so naive against aware is a mismatch).
- `reconcile` keeps the expected side in memory. `reconcile_ordered` and `reconcile_queries` merge two ordered streams in constant memory and **verify the ordering**, raising `ReconciliationError` instead of returning a wrong answer (for example when two engines collate text differently).
- Summaries mask sensitive columns and truncate long values.
- The module is generic. It does not depend on the API module; a later integration phase can feed it API-derived rows.

## Reporting

Database activity is published through the shared-core contracts. There is no second result format.

- `metadata.db_queries`: statement summaries (operation, verb, **fingerprint** with inline literals removed, rows, row count, duration, outcome, `error_kind`), at most 50 per test.
- `metadata.db_connections`: dialect, database *name* (not path, host or user), environment, read-only flag.
- `metadata.db_transactions`: commits, rollbacks and failures, including the rollback at session close.
- Metric `db.query_ms` per statement, on the test's `TestResult`.
- On failure (`capture_diagnostics: on_failure`) a `database/queries.json` artifact with the recent statement summaries is stored in the shared artifact store and cited by the result.
- Never reported: passwords, connection strings, bound parameter values, result rows, host names.

Failures keep the shared taxonomy in `failure.category` (see the table under [Errors and diagnostics](#errors-and-diagnostics)).

## Fixtures and scopes

Load them in the `conftest.py` of database test folders (none is autouse):

```python
import pytest
pytest.importorskip("sqlalchemy")
from complete_automation.database.pytest_plugin import *  # noqa: F401,F403
```

| Fixture | Scope | Provides |
| --- | --- | --- |
| `database_settings` | session | `modules.database`, paths resolved |
| `database_manager` | session | `EngineManager` for the configured database; skips when unconfigured; disposed at session end |
| `db_session_factory` | function | `factory(manager, **options)`; every session is closed after the test, and statements, transactions and diagnostics are reported |
| `db_session` | function | A session on the configured database. Read-only unless `access: read_write`. Rolled back; commit only deliberately |
| `db_readonly_manager` | session | The configured database with engine-enforced read-only |
| `sqlite_database` | function | Manager for a fresh temporary SQLite file, disposed before pytest removes it |
| `sqlite_memory_database` | function | Manager for a fresh in-memory database |
| `example_database`, `example_session` | function | `sqlite_database` with the example schema and seed data committed; a session on it |
| `postgres_database`, `postgres_session` | function | A scratch PostgreSQL schema with the example tables (see below) |

Managers for a configured database are session-scoped so the run shares one pool; sessions are function-scoped so no connection or transaction is shared between tests. Override any fixture by defining one with the same name in a closer `conftest.py`.

### The example schema

`customers` (unique `email`, `NOT NULL` columns, server defaults, a `CHECK` on `status`) ← `orders` (`ON DELETE RESTRICT`, an index, a stored `total`) ← `order_items` (`ON DELETE CASCADE`, a `CHECK` on `quantity`, a stored line `amount`, a composite unique constraint). It is defined with SQLAlchemy Core in `database/example_schema.py`, so it creates on SQLite and PostgreSQL alike. Nothing in the framework depends on it.

## Running the tests

```bash
uv sync --locked --all-extras

uv run pytest tests/unit/database                 # offline unit tests (no database server)
uv run pytest tests/integration/database          # real SQLite databases; PostgreSQL tests skip
uv run pytest tests/examples/database             # the runnable examples
uv run pytest -m database                         # every database test
uv run pytest -m "database and not postgresql"    # everything that needs no external service
uv run pytest tests/integration/database/test_transactions.py::test_savepoints_nest   # one test
uv run pytest -m postgresql -rs                   # PostgreSQL only; skipped with a reason if unconfigured
```

SQLite tests need no server and are deterministic. A skipped PostgreSQL test is reported as **skipped**, never as passed.

## PostgreSQL setup

The PostgreSQL tests are opt-in. They need a **disposable** database and an account that may create and drop schemas in it.

```bash
export CAUTO_MODULES__DATABASE__HOST=127.0.0.1
export CAUTO_MODULES__DATABASE__USERNAME=cauto_tester
export CAUTO_MODULES__DATABASE__PASSWORD=...
uv run pytest -m postgresql --cauto-env postgres_test -rs
```

`config/environments/postgres_test.yaml` names the database (`cauto_test`), grants `read_write`, `allow_ddl` and `allow_destructive`, and lists `cauto_test` as the only privileged database. Each test then creates a scratch schema `cauto_t_<random>`, builds the example tables in it, and drops only that schema afterwards. If the server reports another database than the one configured, the statement is refused.

- **Skipped:** PostgreSQL is not configured, or the configuration does not authorize a scratch schema. The skip reason says which.
- **Error, not skip:** the configuration is complete but the server is unreachable or the credentials are wrong. A configured test that cannot run should not look like a pass.
- Use `127.0.0.1` rather than `localhost` if your system resolves `localhost` to IPv6 first and the server listens only on IPv4; otherwise each connection waits out `connect_timeout_seconds`.
- To test an existing database read-only, use `access: read_only` with a read-only account and the `db_session` fixture.

## Examples

Each example is a test in `tests/examples/database/test_database_examples.py`, so it stays correct as the module changes.

1. **A temporary SQLite database:** `test_example_1_create_a_temporary_sqlite_database`
2. **An example schema:** `test_example_2_define_a_schema_with_sqlalchemy_core`
3. **Insert a record safely:** `test_example_3_insert_a_record_with_bound_parameters`
4. **Query with bound parameters:** `test_example_4_query_with_named_and_positional_parameters`
5. **Validate an expected result:** `test_example_5_validate_an_expected_result`
6. **Verify a rollback:** `test_example_6_verify_that_a_rollback_discards_changes`
7. **Validate schema constraints:** `test_example_7_validate_schema_and_constraints`
8. **Detect duplicates:** `test_example_8_detect_duplicate_records`
9. **Compare datasets:** `test_example_9_compare_expected_and_actual_datasets`
10. **An optional PostgreSQL test:** `test_example_10_optional_postgresql_test`
11. **Publish a result through the shared core:** `test_example_11_publish_a_result_through_the_shared_core_contracts`
12. **Update, verify, roll back:** `test_example_12_update_verify_and_roll_back`

A complete short test, with a rollback check:

```python
def test_an_order_can_be_paid(example_session):
    db = example_session
    db.execute("UPDATE orders SET status = :s WHERE id = :id", {"s": "paid", "id": 1}, expect_rowcount=1)
    assert_record(db, "orders", {"id": 1}, {"status": "paid"})
    db.rollback()
    assert_record(db, "orders", {"id": 1}, {"status": "new"})
```

(`test_example_12_update_verify_and_roll_back` runs this as written.)

## Troubleshooting

- **`refused: INSERT is a write operation but access is read_only`:** the default is read-only. Set `access: read_write` for a database you may change.
- **`refused: ... not in privileged_databases`:** DDL and destructive statements need the server-reported database name in `privileged_databases`.
- **`environment 'prod' is protected`:** protected environments cannot be made writable. Remove the write settings or change `protected_environments` deliberately.
- **`database is locked` / `QueryTimeoutError` on SQLite:** another connection holds a write lock. Commit or roll back the other session, or raise `connect_timeout_seconds` (SQLite's busy timeout).
- **`current transaction is aborted` (PostgreSQL):** a statement failed earlier in the transaction. Contain expected failures in `db.savepoint()` or call `rollback()`.
- **`TransactionError: a transaction is already active`:** `transaction()` was called after statements ran. Call `commit()` or `rollback()` first.
- **A foreign-key test passes on SQLite for no reason:** call `assert_foreign_keys_enforced(db)`. The default `sqlite.foreign_keys: true` enables it.
- **A derived-total assertion fails by `0.00000000000003`:** SQLite `NUMERIC` is floating point. Pass `tolerance=0.005`.
- **`ResultTooLargeError`:** stream with `iterate`, or pass `max_rows`.
- **`the ... records are not ordered by the key` in reconciliation:** order both queries by the key, in the same direction and collation.
- **`psycopg is not installed`:** run `uv sync --extra postgresql`.
- **PermissionError deleting a SQLite file on Windows:** dispose the engine first (`manager.dispose()`), as the fixtures do.

## Known limitations

- Two engines are implemented: SQLite and PostgreSQL. MySQL, SQL Server and Oracle are not.
- Statement classification is keyword-level. A `SELECT` that calls a side-effecting function counts as a read; engine-level read-only mode and database privileges are what stop it (confirmed for `nextval()`/`setval()` on PostgreSQL; advisory locks are not blocked).
- `?` positional placeholders conflict with PostgreSQL's JSONB operators; use named parameters there. Literal colons in SQL text must be written `\:`.
- Sessions are synchronous and single-connection. There is no async API.
- Schema validation does not compare `CHECK` constraints, triggers, sequences or privileges, and skips expression indexes.
- There is no Alembic (or other migration tool) integration beyond reading the version table; none is installed in this repository.
- SQLite cannot reproduce PostgreSQL concurrency, isolation or locking behavior, and has no server-side statement timeout.
- Cross-engine reconciliation of text keys depends on both engines ordering text identically; the ordering check reports when they do not.
- The scratch-schema PostgreSQL fixtures need DDL rights on a disposable database.

## Handoff to the UI phase

The database module is independent of UI automation. A UI test can use it in two ways.

**Indirectly, as test data and verification.** Seed through a session before the browser step and verify state after it:

```python
def test_checkout_creates_an_order(page, example_database):    # page: a future UI fixture
    with example_database.session() as db:
        customer_id = db.scalar("SELECT min(id) FROM customers")
    page.checkout(customer_id)                                    # UI action
    with example_database.session() as db:                        # a new connection: sees only committed data
        assert_row_count(db, "orders", 3, {"customer_id": customer_id})   # 2 seeded + 1 from the UI
```

**Explicitly, through a shared `TestResult`.** Use the same `result_recorder`, `test_context` (and its `correlation_id`) in both modules. The database session records its statements in `metadata.db_queries` and the UI module can add its own metadata, so one result describes the whole workflow.

Guidance for that phase:

- Verify with a **separate connection**, never the one that wrote, so only committed state is checked. UI actions that write need committed data, so use a file-based or configured database, not a rolled-back session.
- A UI test that must not write should use `db_readonly_manager`.
- Keep UI code out of `complete_automation/database/` and database code out of the UI package. Compose them in the test, or in a thin integration layer that imports both.
- Browser-driven writes land in the application's own database account, which is a different concern from the test's read-only verification account. Give the verification account only `SELECT`.
