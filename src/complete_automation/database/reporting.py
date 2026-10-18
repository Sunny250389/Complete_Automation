"""Publishing database activity through the shared-core result and artifact contracts.

Nothing here defines a new result format: queries become JSON ``metadata`` and
a duration ``MetricResult`` on the test's ``TestResult`` (through the core
``ResultRecorder``), and diagnostics become ``ArtifactRef`` files in the shared
``ArtifactStore``. What is reported: the dialect, the database *name* (never
the host, user or password), operation kinds, statement fingerprints with
inline literals removed, durations, row counts and error kinds. Bound
parameter values and result rows are never reported.

Failures keep the shared taxonomy: ``DataAssertionError`` and
``SchemaValidationError`` are assertion failures, unsafe-operation and
configuration problems are ``configuration``, connection and cleanup problems
are ``infrastructure``, timeouts are ``timeout``, and other query errors are
``unknown`` (the test or the SQL is at fault). Query errors also carry their
``kind`` in ``metadata.db_queries[].error_kind``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.database.executor import QuerySummary
from complete_automation.database.transactions import TransactionOutcome

if TYPE_CHECKING:
    from complete_automation.database.engine import EngineManager

MAX_RECORDED_QUERIES = 50


def connection_summary(manager: EngineManager) -> dict[str, Any]:
    """Who the test talked to, without host, user or credentials."""
    name = manager.database
    return {
        "dialect": manager.dialect,
        "database": name if name == ":memory:" else PurePath(name).name,
        "environment": manager.environment,
        "read_only": manager.policy.read_only,
    }


def record_connection(recorder: ResultRecorder, manager: EngineManager) -> None:
    connections = list(recorder.metadata.get("db_connections", []))  # type: ignore[arg-type]
    summary = connection_summary(manager)
    if summary not in connections:
        connections.append(summary)
        recorder.add_metadata(db_connections=connections)


def record_query(recorder: ResultRecorder, summary: QuerySummary) -> None:
    """Append the statement summary to ``metadata.db_queries`` and add a ``db.query_ms`` metric."""
    queries = list(recorder.metadata.get("db_queries", []))  # type: ignore[arg-type]
    if len(queries) < MAX_RECORDED_QUERIES:
        queries.append(summary)
        recorder.add_metadata(db_queries=queries)
        recorder.add_metric(
            "db.query_ms",
            summary["duration_ms"],
            unit="ms",
            metadata={
                "operation": summary["operation"],
                "verb": summary["verb"],
                "outcome": summary["outcome"],
            },
        )


def record_transactions(recorder: ResultRecorder, outcomes: Sequence[TransactionOutcome]) -> None:
    """Add the transaction/savepoint outcomes (committed, rolled_back, failed) to the result metadata."""
    if not outcomes:
        return
    existing = list(recorder.metadata.get("db_transactions", []))  # type: ignore[arg-type]
    existing.extend(
        {"scope": o.scope, "outcome": o.outcome, "duration_ms": o.duration_ms, **({"error": o.error} if o.error else {})}
        for o in outcomes
    )
    recorder.add_metadata(db_transactions=existing[:MAX_RECORDED_QUERIES])


def save_diagnostics(
    store: ArtifactStore,
    manager: EngineManager,
    history: Sequence[QuerySummary],
    outcomes: Sequence[TransactionOutcome],
    *,
    name: str = "queries.json",
) -> ArtifactRef:
    """Store recent statement summaries and transaction outcomes as a JSON artifact under ``database/``."""
    return store.write_json(
        name,
        {
            "connection": connection_summary(manager),
            "queries": list(history),
            "transactions": [o.__dict__ for o in outcomes],
        },
        subdir="database",
        metadata={"dialect": manager.dialect, "queries": len(history)},
    )
