"""Contextual precision checks for retrieval-grounded outputs."""

from deepeval.evaluate import evaluate

from llm_automation.metrics.deepeval_metrics import contextual_precision_metric


def test_contextual_precision_for_dataset(llm_test_cases):
    result = evaluate(test_cases=llm_test_cases, metrics=[contextual_precision_metric()])
    assert all(test_result.success for test_result in result.test_results)