# evaluation/evaluator.py

from metrics import (
    evaluate_relevance,
    evaluate_groundedness,
    evaluate_faithfulness
)

def evaluate_rag_output(query, context, response, expected_answer):
    return {
        "relevance": evaluate_relevance(query, response),
        "groundedness": evaluate_groundedness(context, response),
        "faithfulness": evaluate_faithfulness(context, response),
        "expected_match": expected_answer in response
    }