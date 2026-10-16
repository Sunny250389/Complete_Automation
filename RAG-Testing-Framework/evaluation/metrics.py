# evaluation/metrics.py

from llm_judge import llm_judge

def evaluate_groundedness(context, response):
    prompt = f"""
    Context:
    {context}

    Response:
    {response}

    Is the response fully supported by the context?
    Answer YES or NO with reason.
    """

    return llm_judge(prompt)


def evaluate_relevance(query, response):
    prompt = f"""
    Query: {query}
    Response: {response}

    Is the response relevant to the query?
    """

    return llm_judge(prompt)