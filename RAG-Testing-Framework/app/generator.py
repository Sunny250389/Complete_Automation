# app/generator.py

from openai import OpenAI

client = OpenAI()

def generate_response(query, context):
    prompt = f"""
    Answer the question based ONLY on the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content