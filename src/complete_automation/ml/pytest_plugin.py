"""ML evaluation fixtures, built on the shared-core fixtures.

Load in the ``conftest.py`` of ML test folders::

    pytest.importorskip("sklearn")  # skip cleanly without the "ml" extra
    from complete_automation.ml.pytest_plugin import *  # noqa: F401,F403

=====================  ========  ==============================================================
``ml_settings``        session   ``modules.ml`` settings, paths resolved against the project
``ml_dataset_loader``  session   :class:`DatasetLoader` confined to ``data_root``
``ml_load_baseline``   session   ``load(relative_path)`` -> :class:`Baseline` from ``baseline_dir``
``ml_evaluate``        function  ``evaluate(dataset, model, config, ...)`` -> :class:`EvaluationReport`
=====================  ========  ==============================================================

``ml_evaluate`` runs the evaluator, records metrics and a summary on the test's ``TestResult``
(``result_recorder``) and, unless ``modules.ml.publish_artifacts`` is false, stores the report
artifacts through the shared ``artifact_store``. It returns the report and does **not** raise on a
failed gate, so a test can assert on details; call ``report.assert_passed()`` (or pass
``assert_passed=True``) to turn the verdict into a pytest failure. Override a fixture by defining one
with the same name in a closer ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.config import FrameworkSettings
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.ml.baseline import Baseline
from complete_automation.ml.data import Dataset, DatasetLoader
from complete_automation.ml.evaluator import (
    EvaluationCheck,
    Evaluator,
    RepeatabilityConfig,
    SubgroupConfig,
)
from complete_automation.ml.models import ModelAdapter
from complete_automation.ml.reporting import publish_report, record_result
from complete_automation.ml.schemas import BaselineRule, EvaluationConfig, EvaluationReport, PerturbationSpec
from complete_automation.ml.settings import MLSettings

__all__ = ["ml_dataset_loader", "ml_evaluate", "ml_load_baseline", "ml_settings"]


@pytest.fixture(scope="session")
def ml_settings(framework_settings: FrameworkSettings) -> MLSettings:
    return framework_settings.section("ml", MLSettings).with_base_dir(framework_settings.base_dir)


@pytest.fixture(scope="session")
def ml_dataset_loader(ml_settings: MLSettings) -> DatasetLoader:
    return DatasetLoader(
        ml_settings.data_root, max_rows=ml_settings.max_rows, max_file_bytes=ml_settings.max_file_bytes
    )


@pytest.fixture(scope="session")
def ml_load_baseline(ml_settings: MLSettings) -> Callable[[str | Path], Baseline]:
    def load(relative: str | Path) -> Baseline:
        return Baseline.load(ml_settings.baseline_dir, relative)

    return load


@pytest.fixture
def ml_evaluate(
    ml_settings: MLSettings,
    framework_settings: FrameworkSettings,
    artifact_store: LocalArtifactStore,
    result_recorder: ResultRecorder,
) -> Callable[..., EvaluationReport]:
    def evaluate(
        dataset: Dataset,
        model: ModelAdapter,
        config: EvaluationConfig,
        *,
        baseline: Baseline | None = None,
        baseline_rules: Sequence[BaselineRule] = (),
        robustness: Sequence[PerturbationSpec] = (),
        repeatability: RepeatabilityConfig | None = None,
        subgroups: SubgroupConfig | None = None,
        checks: Sequence[EvaluationCheck] = (),
        assert_passed: bool = False,
    ) -> EvaluationReport:
        report = Evaluator(config).evaluate(
            dataset, model, baseline=baseline, baseline_rules=baseline_rules, robustness=robustness,
            repeatability=repeatability, subgroups=subgroups, checks=checks,
        )
        if ml_settings.publish_artifacts:
            report, _ = publish_report(
                report, artifact_store, recorder=result_recorder, redactor=framework_settings.redactor()
            )
        record_result(result_recorder, report)
        if assert_passed:
            report.assert_passed()
        return report

    return evaluate
