# tests/test_rag.py

from app.rag_pipeline import run_rag
from evaluation.evaluator import evaluate_rag_output
import json


def test_rag_system():
    with open("test_data/dataset.json") as f:
        dataset = json.load(f)

    for case in dataset:
        result = run_rag(case["query"])

        evaluation = evaluate_rag_output(
            case["query"],
            result["context"],
            result["response"],
            case["expected_answer"]
        )

        print(evaluation)