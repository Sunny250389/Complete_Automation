# app/rag_pipeline.py

from retriever import retrieve_context
from generator import generate_response


def run_rag(query: str):
    context = retrieve_context(query)
    response = generate_response(query, context)

    return {
        "query": query,
        "context": context,
        "response": response
    }