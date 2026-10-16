"""DeepEval metric builders used across pytest suites."""

from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric


def answer_relevancy_metric(threshold: float = 0.5) -> AnswerRelevancyMetric:
    return AnswerRelevancyMetric(threshold=threshold)


def contextual_precision_metric(threshold: float = 0.5) -> ContextualPrecisionMetric:
    return ContextualPrecisionMetric(threshold=threshold)