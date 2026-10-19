"""LLM evaluation tests need the "llm" extra (httpx, jsonschema)."""

try:
    import httpx, jsonschema  # noqa: E401,F401
except ImportError:  # the "llm" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.llm_evaluation.pytest_plugin import *  # noqa: F401,F403
