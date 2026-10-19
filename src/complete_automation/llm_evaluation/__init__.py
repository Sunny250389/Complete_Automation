"""LLM evaluation: invoke a model, capture its real response, evaluate it, gate and report.

The model under test is always actually called (Ollama is the supported provider); a judge model is
optional and separate. Importing the package reads no files, opens no connections and configures no
logging. Needs the ``llm`` extra (``httpx``, ``jsonschema``). See ``docs/llm_evaluation.md``.
"""

from complete_automation.llm_evaluation.baseline import Baseline, apply_baseline
from complete_automation.llm_evaluation.clients.base import ModelClient
from complete_automation.llm_evaluation.clients.ollama import ModelAvailability, OllamaClient
from complete_automation.llm_evaluation.datasets import load_dataset, parse_cases, validate_cases
from complete_automation.llm_evaluation.errors import (
    BaselineError,
    EvaluatorError,
    JudgeError,
    LLMConfigurationError,
    LLMDatasetError,
    LLMEvaluationError,
    MalformedResponseError,
    ModelConnectionError,
    ModelHTTPError,
    ModelInvocationError,
    ModelNotFoundError,
    ModelTimeoutError,
    QualityGateError,
)
from complete_automation.llm_evaluation.evaluators.judge import JudgeEvaluator
from complete_automation.llm_evaluation.reporting import publish_report, record_result
from complete_automation.llm_evaluation.runner import EvaluationRunner
from complete_automation.llm_evaluation.schemas import (
    RAG_TEMPLATE,
    BaselineRule,
    CheckSpec,
    CriterionSpec,
    Dataset,
    EvaluationConfig,
    EvaluationRecord,
    GenerationConfig,
    JudgeConfig,
    JudgeResult,
    Message,
    MetricOutcome,
    Outcome,
    ModelConfig,
    ModelRequest,
    ModelResponse,
    PromptTemplate,
    RecordStatus,
    SuiteReport,
    TestCase,
    ThresholdRule,
)
from complete_automation.llm_evaluation.settings import LLMEvaluationSettings

__all__ = [
    "Baseline",
    "BaselineError",
    "BaselineRule",
    "CheckSpec",
    "CriterionSpec",
    "Dataset",
    "EvaluationConfig",
    "EvaluationRecord",
    "EvaluationRunner",
    "EvaluatorError",
    "GenerationConfig",
    "JudgeConfig",
    "JudgeError",
    "JudgeEvaluator",
    "JudgeResult",
    "LLMConfigurationError",
    "LLMDatasetError",
    "LLMEvaluationError",
    "LLMEvaluationSettings",
    "MalformedResponseError",
    "Message",
    "MetricOutcome",
    "ModelAvailability",
    "ModelClient",
    "ModelConfig",
    "ModelConnectionError",
    "ModelHTTPError",
    "ModelInvocationError",
    "ModelNotFoundError",
    "ModelRequest",
    "ModelResponse",
    "ModelTimeoutError",
    "OllamaClient",
    "Outcome",
    "PromptTemplate",
    "QualityGateError",
    "RAG_TEMPLATE",
    "RecordStatus",
    "SuiteReport",
    "TestCase",
    "ThresholdRule",
    "apply_baseline",
    "load_dataset",
    "parse_cases",
    "publish_report",
    "record_result",
    "validate_cases",
]
