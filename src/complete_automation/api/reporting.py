"""Publishing API calls through the shared-core result and artifact contracts.

Nothing here defines a new result format: calls become JSON ``metadata`` and a
latency ``MetricResult`` on the test's ``TestResult`` (via the core
``ResultRecorder``), and exchanges become ``ArtifactRef`` files in the
shared ``ArtifactStore``. Everything is sanitized first.
"""

from __future__ import annotations

import re
from typing import Any

from complete_automation.api.client import ApiResponse
from complete_automation.core.artifacts import ArtifactRef, ArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder

MAX_RECORDED_CALLS = 50


def call_summary(response: ApiResponse, *, environment: str | None = None) -> dict[str, Any]:
    """A small, sanitized, JSON-compatible description of one call."""
    summary: dict[str, Any] = {
        "operation": response.operation,
        "method": response.method,
        "status_code": response.status_code,
        "url": response.url,
        "latency_ms": round(response.latency_ms, 3),
        "total_ms": round(response.total_ms, 3),
        "retries": response.retries,
        "redirects": len(response.redirects),
    }
    if environment is not None:
        summary["environment"] = environment
    return summary


def record_api_call(recorder: ResultRecorder, response: ApiResponse, *, environment: str | None = None) -> None:
    """Append the call to ``metadata.api_calls`` and add an ``api.latency_ms`` metric."""
    calls = list(recorder.metadata.get("api_calls", []))  # type: ignore[arg-type]
    if len(calls) < MAX_RECORDED_CALLS:
        calls.append(call_summary(response, environment=environment))
        recorder.add_metadata(api_calls=calls)
        recorder.add_metric(
            "api.latency_ms",
            round(response.latency_ms, 3),
            unit="ms",
            metadata={"operation": response.operation, "method": response.method, "status_code": response.status_code},
        )


def exchange_name(response: ApiResponse) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", response.operation).strip("_") or "root"
    return f"{response.method.lower()}_{slug}_{response.status_code}.json"


def save_exchange(store: ArtifactStore, response: ApiResponse, *, include_bodies: bool = True) -> ArtifactRef:
    """Store the sanitized request/response pair as a JSON artifact under ``api/``."""
    return store.write_json(
        exchange_name(response),
        response.to_record(include_bodies=include_bodies),
        subdir="api",
        metadata={"operation": response.operation, "method": response.method, "status_code": response.status_code},
    )
