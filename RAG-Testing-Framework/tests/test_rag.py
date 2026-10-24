# tests/test_rag.py
#
# Prototype test, not collected by the framework's pytest run (its testpaths is "tests" at the repo root).
# Run it from this folder. It needs the prototype's dependencies (faiss, openai) and an OpenAI key,
# because the metrics call the LLM judge.

from app.rag_pipeline import run_rag
from evaluation.evaluator import evaluate_rag_output
import json

EXPECTED_KEYS = {"relevance", "groundedness", "faithfulness", "expected_match"}


def test_rag_system():
    with open("test_data/dataset.json") as f:
        dataset = json.load(f)

    for case in dataset:
        result = run_rag(case["query"])
        assert {"context", "response"} <= set(result)

        evaluation = evaluate_rag_output(
            case["query"],
            result["context"],
            result["response"],
            case["expected_answer"]
        )

        assert set(evaluation) == EXPECTED_KEYS
        assert isinstance(evaluation["expected_match"], bool)
        print(evaluation)
