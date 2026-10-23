"""API fixtures, built on the shared-core fixtures.

Load in the ``conftest.py`` of API test folders::

    pytest.importorskip("openapi_spec_validator")  # skip cleanly without the "api" extra (httpx alone is also in "llm")
    from complete_automation.api.pytest_plugin import *  # noqa: F401,F403

Fixtures (none autouse, none touches the network while being set up):

========================  ========  ====================================================
``api_settings``          session   ``modules.api`` settings, file paths resolved
``api_client_factory``    function  builds ``ApiClient`` objects; closes them after the test
``api_client``            function  client for ``modules.api.base_url``; skips if unset
``openapi_contract``      session   contract from ``modules.api.contract``; skips if unset
``mock_api``              function  a fresh :class:`MockApi`
========================  ========  ====================================================

Clients from ``api_client_factory`` record every call in the test's shared
``TestResult`` (``metadata.api_calls`` and an ``api.latency_ms`` metric) and,
per ``modules.api.capture_exchanges``, store sanitized exchanges as artifacts
(default: only when the test fails). Override a fixture by defining one with
the same name in a closer ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from complete_automation.api.client import ApiClient, ApiResponse
from complete_automation.api.contract import OpenApiContract
from complete_automation.api.mocking import MockApi
from complete_automation.api.reporting import record_api_call, save_exchange
from complete_automation.api.settings import ApiSettings
from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.config import FrameworkSettings
from complete_automation.core.context import RunContext
from complete_automation.core.pytest_plugin import ResultRecorder

__all__ = [
    "api_client",
    "api_client_factory",
    "api_settings",
    "mock_api",
    "openapi_contract",
    "pytest_runtest_makereport",
]

MAX_CAPTURED_EXCHANGES = 10
_FAILED = pytest.StashKey[bool]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    report = yield
    if report.when in ("setup", "call") and report.failed:
        item.stash[_FAILED] = True
    return report


@pytest.fixture(scope="session")
def api_settings(framework_settings: FrameworkSettings) -> ApiSettings:
    return framework_settings.section("api", ApiSettings).with_base_dir(framework_settings.base_dir)


@pytest.fixture
def api_client_factory(
    request: pytest.FixtureRequest,
    api_settings: ApiSettings,
    framework_settings: FrameworkSettings,
    run_context: RunContext,
    result_recorder: ResultRecorder,
    artifact_store: LocalArtifactStore,
) -> Iterator[Callable[..., ApiClient]]:
    """``factory(**overrides)`` -> ``ApiClient.from_settings(api_settings, **overrides)``."""
    clients: list[ApiClient] = []
    responses: list[ApiResponse] = []

    def on_response(response: ApiResponse) -> None:
        responses.append(response)
        record_api_call(result_recorder, response, environment=run_context.environment)

    def factory(**overrides: Any) -> ApiClient:
        overrides.setdefault("on_response", on_response)
        overrides.setdefault("redactor", framework_settings.redactor())
        client = ApiClient.from_settings(api_settings, environment=run_context.environment, **overrides)
        clients.append(client)
        return client

    try:
        yield factory
        policy = api_settings.capture_exchanges
        if policy == "always" or (policy == "on_failure" and request.node.stash.get(_FAILED, False)):
            for response in responses[-MAX_CAPTURED_EXCHANGES:]:
                result_recorder.attach(save_exchange(artifact_store, response))
    finally:
        for client in clients:
            client.close()


@pytest.fixture
def api_client(api_settings: ApiSettings, api_client_factory: Callable[..., ApiClient]) -> ApiClient:
    if not api_settings.base_url:
        pytest.skip("modules.api.base_url is not configured (set it in the environment file or CAUTO_MODULES__API__BASE_URL)")
    return api_client_factory()


@pytest.fixture(scope="session")
def openapi_contract(api_settings: ApiSettings, framework_settings: FrameworkSettings) -> OpenApiContract:
    contract = api_settings.contract
    if contract.spec_path is not None:
        return OpenApiContract.from_file(contract.spec_path, redactor=framework_settings.redactor())
    if contract.spec_url is not None:
        return OpenApiContract.from_url(
            contract.spec_url,
            timeout=contract.remote_timeout_seconds,
            verify=api_settings.verify_tls,
            redactor=framework_settings.redactor(),
        )
    pytest.skip("no OpenAPI document configured (modules.api.contract.spec_path or spec_url)")


@pytest.fixture
def mock_api() -> MockApi:
    return MockApi()
