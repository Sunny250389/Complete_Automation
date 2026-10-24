"""Examples of the LLM evaluation module. The offline example runs anywhere; the live one needs an Ollama model."""

try:
    import httpx, jsonschema  # noqa: E401,F401 - both: the "llm" extra
except ImportError:  # the "llm" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.llm_evaluation.pytest_plugin import *  # noqa: F401,F403
