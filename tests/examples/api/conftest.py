"""API tests against the local test service on 127.0.0.1. Needs the "api" extra."""

try:
    import httpx, openapi_spec_validator  # noqa: E401,F401 - both: the "llm" extra installs httpx too
except ImportError:  # the "api" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.api.pytest_plugin import *  # noqa: F401,F403
    from tests.support.api_server import _api_services, local_api, other_origin  # noqa: F401
