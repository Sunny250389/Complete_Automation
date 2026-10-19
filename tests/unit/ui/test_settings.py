"""UiSettings validation and its place in the shared configuration."""

import pytest
from pydantic import ValidationError

from complete_automation.core.config import load_settings
from complete_automation.core.exceptions import ConfigurationError
from complete_automation.ui.browser import context_options, launch_options
from complete_automation.ui.settings import UiSettings

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]


def test_defaults_are_safe_and_cheap() -> None:
    settings = UiSettings()

    assert settings.browser == "chromium" and settings.headless is True
    assert settings.screenshot == "on_failure"
    assert settings.trace == "off" and settings.video == "off"  # expensive diagnostics are opt-in
    assert settings.base_url is None
    assert settings.navigation_timeout_ms == 30000 and settings.action_timeout_ms == 10000
    assert settings.expect_timeout_ms == 5000


@pytest.mark.parametrize("browser", ["chromium", "firefox", "webkit"])
def test_supported_engines(browser: str) -> None:
    assert UiSettings(browser=browser).browser == browser


@pytest.mark.parametrize(
    "values",
    [
        {"browser": "edge"},
        {"base_url": "ftp://host"},
        {"base_url": "localhost:3000"},
        {"base_url": "http://"},
        {"base_url": "http://user:pw@host/"},
        {"navigation_timeout_seconds": 0},
        {"action_timeout_seconds": -1},
        {"expect_timeout_seconds": 0},
        {"viewport": {"width": 10, "height": 720}},
        {"viewport": {"width": 1280}, "unknown": 1},
        {"screenshot": "sometimes"},
        {"trace": "on"},
        {"video": True},
        {"locale": "not a locale"},
        {"timezone_id": "bad zone!"},
        {"test_id_attribute": "9bad"},
        {"launch_args": ("--ok", "two words")},
        {"launch_args": ("rm -rf",)},
        {"slow_mo_ms": -1},
        {"max_screenshots_per_test": 0},
    ],
)
def test_invalid_values_are_rejected_early(values: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValidationError):
        UiSettings(**values)


def test_error_messages_never_echo_credentials() -> None:
    with pytest.raises(ValidationError) as caught:
        UiSettings(base_url="http://user:hunter2@host/")

    assert "hunter2" not in str(caught.value.errors(include_input=False))


def test_settings_are_immutable() -> None:
    with pytest.raises(ValidationError):
        UiSettings().browser = "firefox"  # type: ignore[misc]


def test_launch_and_context_options_follow_the_settings() -> None:
    settings = UiSettings(
        base_url="http://127.0.0.1:8000",
        headless=False,
        launch_args=("--mute-audio",),
        slow_mo_ms=50,
        viewport={"width": 800, "height": 600},
        locale="de_DE",
        timezone_id="Europe/Berlin",
    )

    assert launch_options(settings) == {"headless": False, "args": ["--mute-audio"], "slow_mo": 50}
    options = context_options(settings)
    assert options["viewport"] == {"width": 800, "height": 600}
    assert options["base_url"] == "http://127.0.0.1:8000"
    assert options["locale"] == "de-DE" and options["timezone_id"] == "Europe/Berlin"
    assert "record_video_dir" not in options and options["accept_downloads"] is False


def test_video_directory_is_only_set_when_given(tmp_path) -> None:  # type: ignore[no-untyped-def]
    options = context_options(UiSettings(), video_dir=tmp_path)

    assert options["record_video_dir"] == str(tmp_path)
    assert options["record_video_size"] == options["viewport"]


def _write_environment(config_dir, body: str) -> None:  # type: ignore[no-untyped-def]
    (config_dir / "environments").mkdir()
    (config_dir / "environments" / "dev.yaml").write_text(body, encoding="utf-8")


def test_ui_section_loads_from_shared_configuration(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_environment(
        tmp_path, "modules:\n  ui:\n    browser: firefox\n    headless: false\n    trace: on_failure\n"
    )

    framework = load_settings(config_dir=tmp_path, environ={}, env_file=False)
    ui = framework.section("ui", UiSettings)

    assert (ui.browser, ui.headless, ui.trace) == ("firefox", False, "on_failure")


def test_environment_variables_override_the_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_environment(tmp_path, "modules:\n  ui:\n    browser: firefox\n")

    framework = load_settings(
        config_dir=tmp_path,
        env_file=False,
        environ={"CAUTO_MODULES__UI__BROWSER": "webkit", "CAUTO_MODULES__UI__VIEWPORT__WIDTH": "1000"},
    )
    ui = framework.section("ui", UiSettings)

    assert ui.browser == "webkit" and ui.viewport.width == 1000 and ui.viewport.height == 720


def test_invalid_section_is_a_configuration_error_naming_the_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_environment(tmp_path, "modules:\n  ui:\n    browser: netscape\n")
    framework = load_settings(config_dir=tmp_path, environ={}, env_file=False)

    with pytest.raises(ConfigurationError, match=r"modules\.ui\.browser"):
        framework.section("ui", UiSettings)
