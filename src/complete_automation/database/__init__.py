"""Database testing module: adapters, safe SQL execution, transactions and validation.

Builds on ``complete_automation.core`` for configuration, redaction, results,
reporting and artifacts. Importing this package opens no database connection.
Requires the ``database`` extra (``uv sync --extra database``); PostgreSQL also
needs the ``postgresql`` extra.
"""

from complete_automation.database.adapters import (
    Capabilities,
    DatabaseAdapter,
    PostgreSqlAdapter,
    SqliteAdapter,
    get_adapter_class,
    register_adapter,
)
from complete_automation.database.engine import EngineManager, HealthReport
from complete_automation.database.errors import (
    CleanupError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    DataAssertionError,
    QueryExecutionError,
    QueryTimeoutError,
    ReconciliationError,
    ResultTooLargeError,
    SchemaValidationError,
    TransactionError,
    UnsafeOperationError,
    UnsupportedOperationError,
)
from complete_automation.database.executor import QueryResult, SqlExecutor
from complete_automation.database.reconciliation import (
    ReconciliationConfig,
    ReconciliationResult,
    reconcile,
    reconcile_ordered,
    reconcile_queries,
)
from complete_automation.database.safety import OperationPolicy
from complete_automation.database.session import DatabaseSession
from complete_automation.database.settings import DatabaseSettings, safe_url
from complete_automation.database.sqltext import OperationKind, classify
from complete_automation.database.transactions import TransactionManager, TransactionOutcome

__all__ = [
    "Capabilities",
    "CleanupError",
    "DataAssertionError",
    "DatabaseAdapter",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseError",
    "DatabaseIntegrityError",
    "DatabaseSession",
    "DatabaseSettings",
    "EngineManager",
    "HealthReport",
    "OperationKind",
    "OperationPolicy",
    "PostgreSqlAdapter",
    "QueryExecutionError",
    "QueryResult",
    "QueryTimeoutError",
    "ReconciliationConfig",
    "ReconciliationError",
    "ReconciliationResult",
    "ResultTooLargeError",
    "SchemaValidationError",
    "SqlExecutor",
    "SqliteAdapter",
    "TransactionError",
    "TransactionManager",
    "TransactionOutcome",
    "UnsafeOperationError",
    "UnsupportedOperationError",
    "classify",
    "get_adapter_class",
    "reconcile",
    "reconcile_ordered",
    "reconcile_queries",
    "register_adapter",
    "safe_url",
]
