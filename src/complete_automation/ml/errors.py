"""ML evaluation errors, classified with the shared-core :class:`ErrorCategory`.

Rule for bad input: invalid *data or predictions* never raise from ``Evaluator.evaluate``; they
become blocking findings on the report, so the report always says what was wrong. Invalid
*configuration* (unknown metric, unsupported task, inconsistent options) raises
:class:`MLConfigurationError` before any evaluation runs. ``EvaluationReport.assert_passed``
turns a failed report into the exception that matches the cause.
"""

from __future__ import annotations

from complete_automation.core.exceptions import ErrorCategory, FrameworkError


class MLEvaluationError(FrameworkError):
    """Base class for the ML evaluation module."""


class MLConfigurationError(MLEvaluationError):
    category = ErrorCategory.CONFIGURATION


class DatasetError(MLEvaluationError):
    """The dataset cannot be loaded or violates its contract (fix the data)."""

    category = ErrorCategory.TEST_DATA


class ModelInvocationError(MLEvaluationError):
    """The model raised while predicting. The framework cannot tell a model bug from bad input."""

    category = ErrorCategory.UNKNOWN


class BaselineError(MLEvaluationError):
    category = ErrorCategory.TEST_DATA


class QualityGateError(MLEvaluationError, AssertionError):
    """At least one metric violated its threshold, or a baseline regression rule."""

    category = ErrorCategory.THRESHOLD


class MetricUnavailableError(MLEvaluationError, AssertionError):
    """A gated metric could not be computed (undefined or unsupported); no verdict on the value."""

    category = ErrorCategory.TEST_DATA


class ValidationFailedError(MLEvaluationError, AssertionError):
    """Blocking data or prediction validation errors; metrics were not computed."""

    category = ErrorCategory.TEST_DATA
