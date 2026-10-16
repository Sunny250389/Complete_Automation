# evaluation/llm_judge.py

from openai import OpenAI
client = OpenAI()

def llm_judge(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content