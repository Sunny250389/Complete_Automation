"""API unit tests: everything mocked, no network. Needs the "api" extra."""

try:
    import httpx  # noqa: F401
except ImportError:  # the "api" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.api.pytest_plugin import *  # noqa: F401,F403
