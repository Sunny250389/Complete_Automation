"""LLM evaluation fixtures, built on the shared-core fixtures.

Load in the ``conftest.py`` of LLM-evaluation test folders::

    pytest.importorskip("httpx")  # skip cleanly without the "llm" extra
    from complete_automation.llm_evaluation.pytest_plugin import *  # noqa: F401,F403

========================  ========  ==================================================================
``llm_settings``          session   ``modules.llm_evaluation`` settings, paths resolved to the project
``llm_client``            session   :class:`OllamaClient` (no connection until first use)
``llm_load_dataset``      session   ``load(relative_path)`` -> :class:`Dataset` from ``data_root``
``llm_load_baseline``     session   ``load(relative_path)`` -> :class:`Baseline` from ``baseline_dir``
``llm_live_model``        session   the model under test, or a skip/failure (see below)
``llm_live_judge``        session   the judge model, or a skip/failure
``llm_evaluate``          function  run an evaluation, publish artifacts, record the result
========================  ========  ==================================================================

``llm_live_model`` is how a test asks for a *real* model. It skips unless ``modules.llm_evaluation.live``
is true; when it is, it checks that the endpoint is reachable and the model is installed and skips
with the reason otherwise, or fails if ``require_model`` is true. Nothing here ever substitutes a
mock for the real model. ``llm_evaluate`` does not raise on a failed gate unless ``assert_passed=True``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.config import FrameworkSettings
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.llm_evaluation.baseline import Baseline, apply_baseline
from complete_automation.llm_evaluation.clients.base import ModelClient
from complete_automation.llm_evaluation.clients.ollama import ModelAvailability, OllamaClient
from complete_automation.llm_evaluation.datasets import load_dataset
from complete_automation.llm_evaluation.evaluators.judge import JudgeEvaluator
from complete_automation.llm_evaluation.reporting import publish_report, record_result
from complete_automation.llm_evaluation.runner import EvaluationRunner
from complete_automation.llm_evaluation.schemas import Dataset, EvaluationConfig, ModelConfig, SuiteReport, TestCase
from complete_automation.llm_evaluation.settings import LLMEvaluationSettings

__all__ = [
    "llm_client",
    "llm_evaluate",
    "llm_live_judge",
    "llm_live_model",
    "llm_load_baseline",
    "llm_load_dataset",
    "llm_settings",
]


@pytest.fixture(scope="session")
def llm_settings(framework_settings: FrameworkSettings) -> LLMEvaluationSettings:
    section = framework_settings.section("llm_evaluation", LLMEvaluationSettings)
    return section.with_base_dir(framework_settings.base_dir)


@pytest.fixture(scope="session")
def llm_client() -> Iterator[OllamaClient]:
    client = OllamaClient()
    yield client
    client.close()


@pytest.fixture(scope="session")
def llm_load_dataset(llm_settings: LLMEvaluationSettings) -> Callable[..., Dataset]:
    def load(relative: str | Path, *, version: str | None = None) -> Dataset:
        return load_dataset(llm_settings.data_root, relative, version=version)

    return load


@pytest.fixture(scope="session")
def llm_load_baseline(llm_settings: LLMEvaluationSettings) -> Callable[[str | Path], Baseline]:
    def load(relative: str | Path) -> Baseline:
        return Baseline.load(llm_settings.baseline_dir, relative)

    return load


def _require(settings: LLMEvaluationSettings, client: OllamaClient, name: str | None, what: str) -> ModelAvailability:
    if not settings.live:
        pytest.skip(
            "real-model tests are opt-in: set modules.llm_evaluation.live=true "
            "(CAUTO_MODULES__LLM_EVALUATION__LIVE=true)"
        )
    if not name:
        pytest.skip(f"no {what} configured (modules.llm_evaluation.{'judge_model' if what == 'judge model' else 'model'})")
    judging = what == "judge model"
    base_url = (settings.judge_base_url or settings.base_url) if judging else settings.base_url
    config = ModelConfig(model=name, base_url=base_url, timeout_seconds=settings.timeout_seconds)
    availability = client.check_model(config)
    if not availability.available:
        message = f"{what} {name!r} is not usable: {availability.detail}"
        if settings.require_model:
            pytest.fail(message)
        pytest.skip(message)
    return availability


@pytest.fixture(scope="session")
def llm_live_model(llm_settings: LLMEvaluationSettings, llm_client: OllamaClient) -> ModelAvailability:
    """The configured model under test is reachable and installed (else skip, or fail if required)."""
    return _require(llm_settings, llm_client, llm_settings.model, "model")


@pytest.fixture(scope="session")
def llm_live_judge(llm_settings: LLMEvaluationSettings, llm_client: OllamaClient) -> ModelAvailability:
    return _require(llm_settings, llm_client, llm_settings.judge_model, "judge model")


@pytest.fixture
def llm_evaluate(
    llm_settings: LLMEvaluationSettings,
    framework_settings: FrameworkSettings,
    artifact_store: LocalArtifactStore,
    result_recorder: ResultRecorder,
    run_context,  # noqa: ANN001
    llm_client: OllamaClient,
) -> Callable[..., SuiteReport]:
    def evaluate(
        cases: Dataset | Sequence[TestCase],
        config: EvaluationConfig,
        *,
        client: ModelClient | None = None,
        judge_client: ModelClient | None = None,
        baseline: Baseline | None = None,
        dataset_id: str = "adhoc",
        assert_passed: bool = False,
    ) -> SuiteReport:
        judge = None
        if config.judge is not None:
            judge = JudgeEvaluator(judge_client or client or llm_client, config.judge)
        runner = EvaluationRunner(client or llm_client, config, judge=judge, run_id=run_context.run_id)
        report = runner.run(cases, dataset_id=dataset_id)
        if baseline is not None:
            report = apply_baseline(report, baseline)
        if llm_settings.publish_artifacts:
            report, _ = publish_report(
                report, artifact_store, recorder=result_recorder,
                redactor=framework_settings.redactor(), persist_raw=llm_settings.persist_raw,
            )
        record_result(result_recorder, report)
        if assert_passed:
            report.assert_passed()
        return report

    return evaluate
