"""Replace this with your production LLM or agent entry point."""


def run_llm(input_prompt: str) -> str:
    """Simple deterministic mock app for baseline automation."""
    prompt = input_prompt.lower()
    if "president" in prompt and "united states" in prompt:
        return "Joe Biden"
    if "claude" in prompt:
        return "Anthropic built Claude models."
    if "gpt" in prompt:
        return "OpenAI introduced GPT models."
    return "I need more context to answer accurately."