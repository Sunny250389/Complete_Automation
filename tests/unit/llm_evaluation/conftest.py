"""LLM evaluation tests need the "llm" extra (httpx, jsonschema)."""

try:
    import httpx, jsonschema  # noqa: E401,F401
except ImportError:  # the "llm" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
