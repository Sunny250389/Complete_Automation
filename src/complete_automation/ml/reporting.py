"""Publishing evaluation reports through the shared-core contracts.

Nothing here defines a new result format. The evaluation becomes:

* ``MetricResult`` entries (``ml.<metric>``) and JSON ``metadata`` on the test's ``TestResult``,
  through the core ``ResultRecorder``;
* artifacts in the shared ``ArtifactStore`` under ``ml/``: the full JSON report, a metrics CSV and,
  for classification, a confusion-matrix CSV.

Reports hold metrics, counts, class names and column names, never raw rows or predictions. The JSON
passes through the shared ``Redactor`` before it is stored. Images (for example a confusion-matrix
plot) are not produced: the CSV carries the same information without a plotting dependency.
"""

from __future__ import annotations

import csv
import io

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.redaction import Redactor
from complete_automation.ml.schemas import EvaluationReport

SUBDIR = "ml"


def report_to_dict(report: EvaluationReport, redactor: Redactor | None = None) -> dict:  # type: ignore[type-arg]
    data = report.model_dump(mode="json")
    return redactor.redact(data) if redactor is not None else data


def metrics_csv(report: EvaluationReport) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["metric", "status", "value", "direction", "reason"])
    for item in report.metrics:
        writer.writerow([
            item.name, item.status.value, "" if item.value is None else repr(item.value),
            item.direction.value, item.reason or "",
        ])
    return buffer.getvalue()


def confusion_matrix_csv(report: EvaluationReport) -> str | None:
    for item in report.metrics:
        if item.name == "confusion_matrix" and isinstance(item.detail, dict):
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            labels = item.detail["labels"]
            writer.writerow(["true\\predicted", *labels])  # type: ignore[misc]
            for label, row in zip(labels, item.detail["matrix"]):  # type: ignore[arg-type]
                writer.writerow([label, *row])  # type: ignore[misc]
            return buffer.getvalue()
    return None


def publish_report(
    report: EvaluationReport,
    store: ArtifactStore,
    *,
    recorder: ResultRecorder | None = None,
    redactor: Redactor | None = None,
) -> tuple[EvaluationReport, tuple[ArtifactRef, ...]]:
    """Store the report artifacts; returns the report (with the CSV references) and all references.

    The JSON report lists the CSV artifacts it was published with; it cannot list itself, so the
    JSON reference is only in the returned tuple.
    """
    meta = {"evaluation_id": report.evaluation_id, "task": report.task.value}
    refs = [store.write_text(f"metrics-{report.evaluation_id}.csv", metrics_csv(report),
                             subdir=SUBDIR, media_type="text/csv", metadata=meta)]
    matrix = confusion_matrix_csv(report)
    if matrix is not None:
        refs.append(store.write_text(f"confusion-matrix-{report.evaluation_id}.csv", matrix,
                                     subdir=SUBDIR, media_type="text/csv", metadata=meta))
    published = report.model_copy(update={"artifacts": tuple(refs)})
    refs.append(store.write_json(f"evaluation-{report.evaluation_id}.json",
                                 report_to_dict(published, redactor), subdir=SUBDIR, metadata=meta))
    if recorder is not None:
        for ref in refs:
            recorder.attach(ref)
    return published, tuple(refs)


def record_result(recorder: ResultRecorder, report: EvaluationReport) -> None:
    """Add the evaluation to the current test's result: metrics with their gate verdicts, and a summary."""
    gates = {o.metric: o for o in (report.gates.outcomes if report.gates else ()) if o.required}
    for item in report.metrics:
        gate = gates.get(item.name)
        if item.value is not None:
            recorder.add_metric(
                f"ml.{item.name}", item.value,
                threshold=gate.threshold if gate else None,
                passed=gate.passed if gate else None,
                metadata={"direction": item.direction.value},
            )
        elif item.detail is None:  # undefined/unsupported scalar metric: visible, never a number
            recorder.add_metric(
                f"ml.{item.name}", item.status.value,
                passed=False if gate else None, metadata={"reason": item.reason or ""},
            )
    for name, gate in gates.items():
        if all(m.name != name for m in report.metrics):
            recorder.add_metric(f"ml.{name}", "not_computed", passed=False)
    recorder.add_metadata(ml_evaluation={
        "evaluation_id": report.evaluation_id,
        "task": report.task.value,
        "model": report.model.model_dump(mode="json"),
        "dataset_id": report.dataset.dataset_id,
        "dataset_version": report.dataset.version,
        "n_samples": report.n_samples,
        "n_excluded": report.n_excluded,
        "passed": report.passed,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "failure_reasons": list(report.failure_reasons),
    })
