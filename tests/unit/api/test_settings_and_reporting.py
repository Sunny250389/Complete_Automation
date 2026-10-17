"""ApiSettings validation, client construction from settings, and shared-core reporting integration."""

import json
import ssl
from pathlib import Path

import pytest
from pydantic import ValidationError

from complete_automation.api import ApiClient, ApiKeyAuth, OAuth2ClientCredentials
from complete_automation.api.mocking import MockApi
from complete_automation.api.reporting import call_summary, record_api_call, save_exchange
from complete_automation.api.settings import ApiSettings
from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.config import load_settings
from complete_automation.core.exceptions import ConfigurationError
from complete_automation.core.pytest_plugin import ResultRecorder

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


def test_defaults_are_safe() -> None:
    settings = ApiSettings()

    assert settings.verify_tls is True and settings.follow_redirects is False
    assert settings.retry.enabled is False and "POST" not in settings.retry.methods
    assert all(value > 0 for value in settings.timeouts.model_dump().values())
    assert settings.capture_exchanges == "on_failure" and settings.log_bodies is False


def test_insecure_tls_needs_two_explicit_settings() -> None:
    with pytest.raises(ValidationError, match="allow_insecure_tls"):
        ApiSettings(verify_tls=False)
    assert ApiSettings(verify_tls=False, allow_insecure_tls=True).verify_tls is False


@pytest.mark.parametrize("value", ["ftp://x", "api.example.test", "javascript:alert(1)"])
def test_base_url_must_be_http(value: str) -> None:
    with pytest.raises(ValidationError, match="base_url"):
        ApiSettings(base_url=value)


def test_protected_environments_are_read_only() -> None:
    settings = ApiSettings()
    assert settings.is_read_only("prod") and settings.is_read_only("Production")
    assert not settings.is_read_only("dev")
    assert ApiSettings(read_only=False).is_read_only("prod") is False  # explicit opt-in to mutate production
    assert ApiSettings(read_only=True).is_read_only("dev") is True


def test_settings_load_through_the_shared_core(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    (config_dir / "environments").mkdir(parents=True)
    (config_dir / "environments" / "dev.yaml").write_text(
        "modules:\n  api:\n    base_url: https://api.example.test/v1\n"
        "    timeouts: {read: 12}\n    auth: {type: api_key, name: X-Key}\n"
        "    contract: {spec_path: contracts/users.yaml}\n",
        encoding="utf-8",
    )
    framework = load_settings(
        config_dir=config_dir, env_file=False, environ={"CAUTO_MODULES__API__AUTH__VALUE": "k-from-env"}
    )

    api = framework.section("api", ApiSettings).with_base_dir(framework.base_dir)

    assert api.timeouts.read == 12 and api.auth.value.get_secret_value() == "k-from-env"  # type: ignore[union-attr]
    assert api.contract.spec_path == (tmp_path / "contracts" / "users.yaml").resolve()
    assert "k-from-env" not in repr(framework) and "k-from-env" not in repr(api)


def test_client_from_settings() -> None:
    settings = ApiSettings(
        base_url="https://api.example.test",
        timeouts={"connect": 1, "read": 2, "write": 3, "pool": 4},
        auth={"type": "api_key", "value": "k", "name": "X-Key"},
        retry={"enabled": True, "max_attempts": 2},
        default_headers={"X-Client": "suite"},
    )
    with ApiClient.from_settings(settings, environment="dev") as client:
        timeout = client.httpx_client.timeout
        assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (1, 2, 3, 4)
        assert isinstance(client.httpx_client.auth, ApiKeyAuth)
        assert client.retry is not None and client.retry.max_attempts == 2
        assert client.httpx_client.headers["X-Client"] == "suite"
        assert client.sanitizer.is_sensitive("X-Key")
        assert client.read_only is False
    with ApiClient.from_settings(settings, environment="prod") as protected:
        assert protected.read_only is True
    with pytest.raises(ConfigurationError, match="base_url is not set"):
        ApiClient.from_settings(ApiSettings(), environment="dev")


def test_ca_bundle_builds_an_ssl_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = {}
    monkeypatch.setattr(ssl, "create_default_context", lambda cafile=None: seen.setdefault("cafile", cafile) and ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    bundle = tmp_path / "ca.pem"
    ApiClient.from_settings(ApiSettings(base_url="https://h.test", ca_bundle=bundle), environment="dev").close()
    assert seen["cafile"] == str(bundle)


def test_oauth2_settings_build_a_strategy() -> None:
    settings = ApiSettings(base_url="https://h.test", auth={
        "type": "oauth2_client_credentials", "token_url": "https://idp.test/token",
        "client_id": "c", "client_secret": "s", "scopes": ["read"]})
    with ApiClient.from_settings(settings, environment="dev") as client:
        assert isinstance(client.httpx_client.auth, OAuth2ClientCredentials)


def test_reporting_helpers_use_shared_contracts(mock_api: MockApi, tmp_path: Path) -> None:
    mock_api.route("POST", "/login").respond(401, json={"error": "bad credentials", "token": "t-1"})
    with mock_api.client() as client:
        response = client.post("/login", json={"password": "pw-1"}, params={"api_key": "k-1"})

    recorder = ResultRecorder()
    record_api_call(recorder, response, environment="dev")
    record_api_call(recorder, response, environment="dev")
    ref = save_exchange(LocalArtifactStore(tmp_path), response)

    calls = recorder.metadata["api_calls"]
    assert len(calls) == 2 and calls[0] == call_summary(response, environment="dev")
    assert calls[0]["status_code"] == 401 and calls[0]["operation"] == "/login"
    assert [m.name for m in recorder.metrics] == ["api.latency_ms", "api.latency_ms"]
    assert ref.path == "api/post_login_401.json" and ref.media_type == "application/json"
    stored = (tmp_path / ref.path).read_text(encoding="utf-8")
    for secret in ("pw-1", "k-1", "t-1"):
        assert secret not in stored and secret not in json.dumps(calls)


def test_fixtures_publish_api_results_through_the_core_plugins(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a failing API test's result carries api_calls metadata and an exchange artifact."""
    for name in ("CAUTO_ENVIRONMENT", "CAUTO_EXECUTION__RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    config = pytester.path / "config"
    (config / "environments").mkdir(parents=True)
    (config / "environments" / "dev.yaml").write_text("modules:\n  api:\n    base_url: https://api.example.test\n", encoding="utf-8")
    monkeypatch.setenv("CAUTO_CONFIG_DIR", str(config))
    pytester.makeconftest(
        'pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.api.pytest_plugin"]\n'
    )
    pytester.makepyfile(
        """
        import httpx, pytest
        from complete_automation.api.assertions import assert_status

        def handler(request):
            return httpx.Response(500 if request.url.path == "/fail" else 200, json={"password": "pw-9"})

        @pytest.mark.cauto_module("api")
        def test_passes(api_client_factory):
            client = api_client_factory(transport=httpx.MockTransport(handler))
            assert_status(client.get("/ok", params={"token": "q-9"}), 200)

        @pytest.mark.cauto_module("api")
        def test_fails(api_client_factory):
            client = api_client_factory(transport=httpx.MockTransport(handler))
            assert_status(client.get("/fail"), 200)

        def test_api_client_uses_settings(api_client):
            assert str(api_client.base_url) == "https://api.example.test"
            assert not api_client.is_closed
        """
    )

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--cauto-report")
    result.assert_outcomes(passed=2, failed=1)

    report_file = next((pytester.path / "reports").glob("*/results.json"))
    document = json.loads(report_file.read_text(encoding="utf-8"))
    results = {r["name"]: r for r in document["results"]}
    passed, failed = results["test_passes"], results["test_fails"]

    assert passed["metadata"]["api_calls"][0]["status_code"] == 200
    assert passed["artifacts"] == []  # exchanges captured only on failure by default
    assert failed["status"] == "failed" and failed["failure"]["category"] == "assertion"
    assert "expected status 200, got 500" in failed["failure"]["message"]
    assert failed["metadata"]["api_calls"][0]["operation"] == "/fail"
    assert failed["metrics"][0]["name"] == "api.latency_ms"
    artifact = failed["artifacts"][0]
    stored = next((pytester.path / "artifacts").glob(f"*/{artifact['path']}")).read_text(encoding="utf-8")
    assert json.loads(stored)["response"]["status_code"] == 500
    assert "pw-9" not in report_file.read_text(encoding="utf-8") + stored
    assert "q-9" not in report_file.read_text(encoding="utf-8")
