"""ML model evaluation: datasets, predictions, metrics, quality gates, baselines and reports.

Evaluation only: nothing here trains, tunes or serves a model. Needs the ``ml`` extra
(``numpy``, ``pandas``, ``scikit-learn``). Importing the package reads no files and opens no
connections. See ``docs/ml_evaluation.md``.
"""

from complete_automation.ml.baseline import Baseline, compare_to_baseline
from complete_automation.ml.data import Dataset, DatasetLoader
from complete_automation.ml.errors import (
    BaselineError,
    DatasetError,
    MetricUnavailableError,
    MLConfigurationError,
    MLEvaluationError,
    ModelInvocationError,
    QualityGateError,
    ValidationFailedError,
)
from complete_automation.ml.evaluator import (
    CheckContext,
    EvaluationCheck,
    Evaluator,
    RepeatabilityConfig,
    SubgroupConfig,
)
from complete_automation.ml.models import (
    EstimatorAdapter,
    FunctionAdapter,
    ModelAdapter,
    PrecomputedAdapter,
    Predictions,
)
from complete_automation.ml.schemas import (
    BaselineRule,
    ColumnSpec,
    DatasetSchema,
    EvaluationConfig,
    EvaluationReport,
    PerturbationSpec,
    TaskType,
    ThresholdRule,
)
from complete_automation.ml.thresholds import evaluate_gates, parse_thresholds

__all__ = [
    "Baseline",
    "BaselineError",
    "BaselineRule",
    "CheckContext",
    "ColumnSpec",
    "Dataset",
    "DatasetError",
    "DatasetLoader",
    "DatasetSchema",
    "EstimatorAdapter",
    "EvaluationCheck",
    "EvaluationConfig",
    "EvaluationReport",
    "Evaluator",
    "FunctionAdapter",
    "MLConfigurationError",
    "MLEvaluationError",
    "MetricUnavailableError",
    "ModelAdapter",
    "ModelInvocationError",
    "PerturbationSpec",
    "PrecomputedAdapter",
    "Predictions",
    "QualityGateError",
    "RepeatabilityConfig",
    "SubgroupConfig",
    "TaskType",
    "ThresholdRule",
    "ValidationFailedError",
    "compare_to_baseline",
    "evaluate_gates",
    "parse_thresholds",
]
