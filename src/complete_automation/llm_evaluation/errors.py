"""LLM evaluation errors, classified with the shared-core :class:`ErrorCategory`.

Three kinds of failure stay apart, because they mean different things:

* the **model call** failed (``ModelInvocationError`` and subclasses): no response exists;
* an **evaluator** failed (``EvaluatorError``, ``JudgeError``): a response exists but no verdict
  could be reached;
* a **threshold** was missed (``QualityGateError``): the model's output is not good enough.
"""

from __future__ import annotations

from complete_automation.core.exceptions import ErrorCategory, FrameworkError, TestDataError


class LLMEvaluationError(FrameworkError):
    """Base class for the LLM evaluation module."""


class LLMConfigurationError(LLMEvaluationError):
    category = ErrorCategory.CONFIGURATION


class LLMDatasetError(LLMEvaluationError, TestDataError):
    """The dataset cannot be loaded or violates its contract (fix the data)."""

    category = ErrorCategory.TEST_DATA


class ModelInvocationError(LLMEvaluationError):
    """Calling the model failed. Nothing was generated."""

    category = ErrorCategory.INFRASTRUCTURE
    error_kind = "invocation"


class ModelConnectionError(ModelInvocationError):
    error_kind = "connection"


class ModelTimeoutError(ModelInvocationError):
    category = ErrorCategory.TIMEOUT
    error_kind = "timeout"


class ModelHTTPError(ModelInvocationError):
    error_kind = "http"

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelNotFoundError(ModelHTTPError):
    error_kind = "model_not_found"


class MalformedResponseError(ModelInvocationError):
    """The server answered, but not with the documented response structure."""

    error_kind = "malformed_response"


class EvaluatorError(LLMEvaluationError):
    """An evaluator could not produce a verdict (not the same as the response failing a check)."""

    category = ErrorCategory.UNKNOWN


class JudgeError(EvaluatorError):
    """The judge model failed or returned output that cannot be used."""


class QualityGateError(LLMEvaluationError, AssertionError):
    """A metric violated its threshold or a baseline rule."""

    category = ErrorCategory.THRESHOLD


class BaselineError(LLMEvaluationError):
    category = ErrorCategory.TEST_DATA
