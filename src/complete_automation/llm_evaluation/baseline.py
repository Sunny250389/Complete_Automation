"""Baseline files: aggregate metrics of an earlier run, compared with the current one.

A baseline holds numbers only (no prompts or responses). A changed value is reported but never a
failure by itself; only a :class:`BaselineRule` (an explicit allowance for getting worse) can fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore
from complete_automation.core.datasets import DataLoader
from complete_automation.core.exceptions import TestDataError
from complete_automation.llm_evaluation import gates
from complete_automation.llm_evaluation.errors import BaselineError
from complete_automation.llm_evaluation.schemas import SuiteReport


class Baseline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    baseline_id: str = Field(min_length=1)
    dataset_id: str
    dataset_fingerprint: str | None = None
    model: str
    template: str | None = None
    aggregates: dict[str, float | None]

    @classmethod
    def from_report(cls, report: SuiteReport, baseline_id: str | None = None) -> Baseline:
        return cls(
            baseline_id=baseline_id or f"{report.config.model.model}@{report.dataset_id}",
            dataset_id=report.dataset_id,
            dataset_fingerprint=report.dataset_fingerprint,
            model=report.config.model.model,
            template=report.config.template.label if report.config.template else None,
            aggregates=dict(report.aggregates),
        )

    def save(self, store: ArtifactStore, name: str | None = None) -> ArtifactRef:
        """Write the baseline as an artifact; copy it into the baseline directory to adopt it."""
        return store.write_json(
            name or f"baseline-{self.baseline_id}.json", self.model_dump(mode="json"), subdir="llm_evaluation"
        )

    @classmethod
    def load(cls, root: Path, relative: str | Path) -> Baseline:
        try:
            data = DataLoader(root).load(relative)
        except TestDataError as error:
            raise BaselineError(str(error)) from error
        try:
            return cls.model_validate(data)
        except ValueError as error:
            raise BaselineError(f"invalid baseline file {relative}: {error}") from error


def apply_baseline(report: SuiteReport, baseline: Baseline) -> SuiteReport:
    """Return ``report`` with its baseline comparison filled in (rules: ``config.baseline_rules``)."""
    if baseline.dataset_id != report.dataset_id:
        raise BaselineError(
            f"baseline {baseline.baseline_id!r} is for dataset {baseline.dataset_id!r}, "
            f"this run used {report.dataset_id!r}"
        )
    warnings = list(report.warnings)
    if baseline.dataset_fingerprint and report.dataset_fingerprint != baseline.dataset_fingerprint:
        warnings.append("the dataset content changed since the baseline was recorded")
    if baseline.model != report.config.model.model:
        warnings.append(f"baseline model {baseline.model!r} differs from {report.config.model.model!r}")
    comparison = gates.compare_to_baseline(
        baseline.baseline_id, baseline.aggregates, report.aggregates, report.config.baseline_rules
    )
    return report.model_copy(update={"baseline": comparison, "warnings": tuple(warnings)})
