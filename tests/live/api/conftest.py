"""Opt-in tests against a real external API; they skip unless modules.api is configured."""

try:
    import httpx, openapi_spec_validator  # noqa: E401,F401 - both: the "llm" extra installs httpx too
except ImportError:  # the "api" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.api.pytest_plugin import *  # noqa: F401,F403
