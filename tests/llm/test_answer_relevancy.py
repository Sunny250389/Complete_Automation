"""Answer relevancy regression suite."""

import pytest
from deepeval.evaluate import evaluate

from complete_automation.metrics.deepeval_metrics import answer_relevancy_metric

pytestmark = pytest.mark.live


def test_answer_relevancy_for_dataset(llm_test_cases):
    result = evaluate(test_cases=llm_test_cases, metrics=[answer_relevancy_metric()])
    assert all(test_result.success for test_result in result.test_results)