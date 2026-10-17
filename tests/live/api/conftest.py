"""Opt-in tests against a real external API; they skip unless modules.api is configured."""

try:
    import httpx  # noqa: F401
except ImportError:  # the "api" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.api.pytest_plugin import *  # noqa: F401,F403
