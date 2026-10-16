from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from complete_automation.core.config import FrameworkSettings, load_settings
from complete_automation.core.exceptions import ConfigurationError

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "config"
    (directory / "environments").mkdir(parents=True)
    return directory


def write_env(config_dir: Path, name: str, text: str) -> None:
    (config_dir / "environments" / f"{name}.yaml").write_text(text, encoding="utf-8")


def load(config_dir: Path, **kwargs) -> FrameworkSettings:
    kwargs.setdefault("env_file", False)
    kwargs.setdefault("environ", {})
    return load_settings(config_dir=config_dir, **kwargs)


class ApiSettings(BaseModel):
    base_url: str
    timeout_seconds: float = 5.0
    token: SecretStr | None = None


def test_defaults_load_without_any_files(config_dir: Path) -> None:
    settings = load(config_dir)

    assert settings.environment == "dev"
    assert settings.logging.level == "INFO"
    assert settings.retry.max_attempts == 3
    assert settings.reporting.json_enabled is False
    assert settings.base_dir == config_dir.parent.resolve()
    assert settings.artifacts.root == config_dir.parent.resolve() / "artifacts"


def test_environment_file_overrides_defaults(config_dir: Path) -> None:
    write_env(config_dir, "test", "logging:\n  level: debug\nreporting:\n  json_enabled: true\n")

    settings = load(config_dir, environment="test")

    assert settings.environment == "test"
    assert settings.logging.level == "DEBUG"
    assert settings.reporting.json_enabled is True
    assert settings.retry.max_attempts == 3  # untouched default


def test_process_environment_beats_file(config_dir: Path) -> None:
    write_env(config_dir, "dev", "logging:\n  level: WARNING\n")

    settings = load(config_dir, environ={"CAUTO_LOGGING__LEVEL": "error"})

    assert settings.logging.level == "ERROR"


def test_dotenv_sits_between_file_and_process_environment(config_dir: Path) -> None:
    write_env(config_dir, "dev", "retry:\n  max_attempts: 2\nlogging:\n  level: WARNING\n")
    (config_dir.parent / ".env").write_text(
        "CAUTO_RETRY__MAX_ATTEMPTS=5\nCAUTO_LOGGING__LEVEL=DEBUG\nOTHER_TOOL=ignored\n",
        encoding="utf-8",
    )

    settings = load(config_dir, env_file=True, environ={"CAUTO_LOGGING__LEVEL": "ERROR"})

    assert settings.retry.max_attempts == 5  # .env beats the file
    assert settings.logging.level == "ERROR"  # process env beats .env


def test_overrides_beat_every_other_source(config_dir: Path) -> None:
    write_env(config_dir, "dev", "logging:\n  level: WARNING\n")

    settings = load(
        config_dir,
        environ={"CAUTO_LOGGING__LEVEL": "ERROR"},
        overrides={"logging": {"level": "CRITICAL"}},
    )

    assert settings.logging.level == "CRITICAL"


def test_environment_argument_beats_environment_variable(config_dir: Path) -> None:
    write_env(config_dir, "dev", "")
    write_env(config_dir, "test", "")

    assert load(config_dir, environment="dev", environ={"CAUTO_ENVIRONMENT": "test"}).environment == "dev"
    assert load(config_dir, environ={"CAUTO_ENVIRONMENT": "test"}).environment == "test"


def test_unknown_explicit_environment_is_an_error(config_dir: Path) -> None:
    write_env(config_dir, "dev", "")
    write_env(config_dir, "test", "")

    with pytest.raises(ConfigurationError, match=r"unknown environment 'prod'.*available: dev, test"):
        load(config_dir, environment="prod")


@pytest.mark.parametrize("name", ["../etc", "Prod", "a/b", ""])
def test_invalid_environment_names_are_rejected(config_dir: Path, name: str) -> None:
    with pytest.raises(ConfigurationError):
        load(config_dir, environment=name or None, environ={"CAUTO_ENVIRONMENT": name})


def test_invalid_type_names_the_field_but_not_the_value(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError) as raised:
        load(config_dir, environ={"CAUTO_RETRY__MAX_ATTEMPTS": "many-s3cr3t"})

    message = str(raised.value)
    assert "retry.max_attempts" in message
    assert "many-s3cr3t" not in message


def test_out_of_range_value_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError, match="retry.max_attempts"):
        load(config_dir, overrides={"retry": {"max_attempts": 0}})


def test_unknown_keys_are_rejected_not_ignored(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError, match="loggng"):
        load(config_dir, environ={"CAUTO_LOGGNG__LEVEL": "DEBUG"})


def test_malformed_yaml_is_reported(config_dir: Path) -> None:
    write_env(config_dir, "dev", "logging: [unclosed\n")

    with pytest.raises(ConfigurationError, match="malformed YAML") as raised:
        load(config_dir)
    assert raised.value.__cause__ is not None


def test_yaml_must_be_a_mapping(config_dir: Path) -> None:
    write_env(config_dir, "dev", "- just\n- a list\n")

    with pytest.raises(ConfigurationError, match="mapping"):
        load(config_dir)


def test_yaml_is_loaded_safely(config_dir: Path) -> None:
    write_env(config_dir, "dev", "logging: !!python/object/apply:os.system ['echo unsafe']\n")

    with pytest.raises(ConfigurationError, match="malformed YAML"):
        load(config_dir)


def test_json_values_in_environment_variables(config_dir: Path) -> None:
    settings = load(config_dir, environ={"CAUTO_SECURITY__REDACT_KEYS": '["session_id", "pin"]'})

    assert settings.security.redact_keys == ("session_id", "pin")


def test_malformed_json_variable_is_an_error(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not parse"):
        load(config_dir, environ={"CAUTO_SECURITY__REDACT_KEYS": "[broken"})


def test_missing_explicit_env_file_is_an_error(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError, match=".env file not found"):
        load(config_dir, env_file=config_dir / "missing.env")


def test_module_section_reports_missing_required_values(config_dir: Path) -> None:
    settings = load(config_dir)

    with pytest.raises(ConfigurationError, match=r"modules\.api\.base_url: Field required"):
        settings.section("api", ApiSettings)


def test_module_section_reads_file_and_environment(config_dir: Path) -> None:
    write_env(config_dir, "dev", "modules:\n  api:\n    base_url: https://example.test\n")

    settings = load(config_dir, environ={"CAUTO_MODULES__API__TIMEOUT_SECONDS": "12.5"})
    api = settings.section("api", ApiSettings)

    assert api.base_url == "https://example.test"
    assert api.timeout_seconds == 12.5


def test_secrets_never_appear_in_representations(config_dir: Path) -> None:
    settings = load(
        config_dir,
        environ={
            "CAUTO_MODULES__API__BASE_URL": "https://user:url-pass-123@example.test",
            "CAUTO_MODULES__API__TOKEN": "tok-s3cr3t-456",
        },
    )

    for text in (repr(settings), str(settings), repr(settings.redacted_dump())):
        assert "tok-s3cr3t-456" not in text
        assert "url-pass-123" not in text
    assert settings.section("api", ApiSettings).token.get_secret_value() == "tok-s3cr3t-456"
    assert "tok-s3cr3t-456" not in repr(settings.section("api", ApiSettings))


def test_secret_values_are_not_echoed_in_section_errors(config_dir: Path) -> None:
    settings = load(
        config_dir,
        environ={
            "CAUTO_MODULES__API__BASE_URL": "https://example.test",
            "CAUTO_MODULES__API__TIMEOUT_SECONDS": "s3cr3t-not-a-number",
        },
    )

    with pytest.raises(ConfigurationError) as raised:
        settings.section("api", ApiSettings)
    assert "s3cr3t-not-a-number" not in str(raised.value)


def test_explicit_config_dir_is_independent_of_working_directory(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    write_env(config_dir, "dev", "artifacts:\n  root: out/artifacts\n")

    from_argument = load(config_dir)
    from_variable = load_settings(env_file=False, environ={"CAUTO_CONFIG_DIR": str(config_dir)})

    expected = config_dir.parent.resolve() / "out" / "artifacts"
    assert from_argument.artifacts.root == expected
    assert from_variable.artifacts.root == expected


def test_absolute_paths_are_kept(config_dir: Path, tmp_path: Path) -> None:
    target = (tmp_path / "abs-reports").resolve()

    settings = load(config_dir, overrides={"reporting": {"output_dir": str(target)}})

    assert settings.reporting.output_dir == target


def test_settings_are_immutable(config_dir: Path) -> None:
    settings = load(config_dir)

    with pytest.raises(ValidationError):
        settings.environment = "prod"  # type: ignore[misc]


def test_repository_environment_files_are_valid() -> None:
    dev = load_settings(config_dir=REPO_CONFIG, environment="dev", env_file=False, environ={})
    test = load_settings(config_dir=REPO_CONFIG, environment="test", env_file=False, environ={})

    assert dev.reporting.json_enabled is False
    assert test.reporting.json_enabled is True
    assert test.logging.format == "json"
