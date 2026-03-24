"""Answer relevancy regression suite."""

from deepeval.evaluate import evaluate

from llm_automation.metrics.deepeval_metrics import answer_relevancy_metric


def test_answer_relevancy_for_dataset(llm_test_cases):
    result = evaluate(test_cases=llm_test_cases, metrics=[answer_relevancy_metric()])
    assert all(test_result.success for test_result in result.test_results)