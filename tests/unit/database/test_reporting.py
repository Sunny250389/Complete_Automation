"""Publishing database activity through the shared-core ResultRecorder and artifact store."""

import json
from pathlib import Path

import pytest

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.database.engine import EngineManager
from complete_automation.database.reporting import (
    MAX_RECORDED_QUERIES,
    connection_summary,
    record_connection,
    record_query,
    record_transactions,
    save_diagnostics,
)
from complete_automation.database.sandbox import sqlite_sandbox_settings
from complete_automation.database.transactions import TransactionOutcome

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]


def _summary(n: int = 0) -> dict[str, object]:
    return {
        "outcome": "ok", "operation": "read", "verb": "SELECT", "statement": f"SELECT {n} FROM t WHERE a = ?",
        "rows": 1, "rowcount": None, "duration_ms": 1.5, "dialect": "sqlite",
    }  # fmt: skip


def test_queries_become_metadata_and_a_latency_metric() -> None:
    recorder = ResultRecorder()

    record_query(recorder, _summary())  # type: ignore[arg-type]

    assert recorder.metadata["db_queries"][0]["statement"] == "SELECT 0 FROM t WHERE a = ?"  # type: ignore[index]
    metric = recorder.metrics[0]
    assert (metric.name, metric.value, metric.unit) == ("db.query_ms", 1.5, "ms")
    assert metric.metadata == {"operation": "read", "verb": "SELECT", "outcome": "ok"}


def test_recorded_queries_are_bounded() -> None:
    recorder = ResultRecorder()

    for n in range(MAX_RECORDED_QUERIES + 25):
        record_query(recorder, _summary(n))  # type: ignore[arg-type]

    assert len(recorder.metadata["db_queries"]) == MAX_RECORDED_QUERIES  # type: ignore[arg-type]


def test_transaction_outcomes_are_recorded_without_errors_leaking_detail() -> None:
    recorder = ResultRecorder()

    record_transactions(recorder, [TransactionOutcome("transaction", "committed", 2.0), TransactionOutcome("savepoint", "failed", 1.0, "OperationalError")])
    record_transactions(recorder, [])

    transactions = recorder.metadata["db_transactions"]
    assert [t["outcome"] for t in transactions] == ["committed", "failed"]  # type: ignore[index]
    assert transactions[1]["error"] == "OperationalError"  # type: ignore[index]


def test_connection_summaries_hold_no_host_user_path_or_password(tmp_path: Path) -> None:
    manager = EngineManager(sqlite_sandbox_settings(tmp_path / "secret_dir" / "app.db"), environment="ci")
    recorder = ResultRecorder()

    record_connection(recorder, manager)
    record_connection(recorder, manager)  # recorded once

    assert recorder.metadata["db_connections"] == [
        {"dialect": "sqlite", "database": "app.db", "environment": "ci", "read_only": False}
    ]
    assert "secret_dir" not in json.dumps(connection_summary(manager))


def test_diagnostics_are_stored_as_a_shared_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    manager = EngineManager(sqlite_sandbox_settings(":memory:"))

    ref = save_diagnostics(store, manager, [_summary()], [TransactionOutcome("transaction", "rolled_back", 0.5)])  # type: ignore[list-item]

    assert ref.path == "database/queries.json" and ref.media_type == "application/json"
    document = json.loads(store.resolve(ref).read_text(encoding="utf-8"))
    assert document["connection"]["dialect"] == "sqlite"
    assert document["queries"][0]["statement"].endswith("a = ?")
    assert document["transactions"][0]["outcome"] == "rolled_back"
