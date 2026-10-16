"""Typed framework configuration and its loader.

Precedence, lowest to highest:

1. Defaults declared on the models below.
2. ``<config_dir>/environments/<environment>.yaml``.
3. ``CAUTO_*`` values from a local ``.env`` file (``<base_dir>/.env``).
4. ``CAUTO_*`` process environment variables.
5. ``overrides`` passed to :func:`load_settings`.

Environment variables map to nested keys with ``__``:
``CAUTO_LOGGING__LEVEL=DEBUG`` sets ``logging.level``. A value starting with
``[`` or ``{`` is parsed as JSON. Domain modules keep their settings under
``modules.<name>`` and read them with :meth:`FrameworkSettings.section`.

Relative paths in the configuration resolve against ``base_dir``, the parent of
the configuration directory, so results do not depend on the working directory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from complete_automation.core.exceptions import ConfigurationError
from complete_automation.core.paths import find_project_root
from complete_automation.core.redaction import Redactor

ENV_PREFIX = "CAUTO_"
CONFIG_DIR_VARIABLE = f"{ENV_PREFIX}CONFIG_DIR"
DEFAULT_ENVIRONMENT = "dev"

SectionT = TypeVar("SectionT", bound=BaseModel)


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["text", "json"] = "text"
    console: bool = True
    file: Path | None = None
    """Optional log file; relative paths resolve against ``base_dir``."""

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class ReportingSettings(_Section):
    output_dir: Path = Path("reports")
    json_enabled: bool = False


class ArtifactSettings(_Section):
    root: Path = Path("artifacts")


class ExecutionSettings(_Section):
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    """Fixed run ID (for example shared by parallel workers); generated when unset."""
    default_timeout_seconds: float = Field(default=30.0, gt=0)


class RetrySettings(_Section):
    max_attempts: int = Field(default=3, ge=1, le=20)
    delay_seconds: float = Field(default=0.5, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    max_delay_seconds: float = Field(default=10.0, ge=0)
    jitter_ratio: float = Field(default=0.0, ge=0, le=1)


class SecuritySettings(_Section):
    redact_keys: tuple[str, ...] = ()
    """Extra key names to mask, in addition to the built-in sensitive names."""
    redaction_mask: str = Field(default="***", min_length=1)

    def redactor(self) -> Redactor:
        return Redactor(extra_keys=self.redact_keys, mask=self.redaction_mask)


class FrameworkSettings(_Section):
    environment: str = Field(default=DEFAULT_ENVIRONMENT, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_dir: Path
    logging: LoggingSettings = LoggingSettings()
    reporting: ReportingSettings = ReportingSettings()
    artifacts: ArtifactSettings = ArtifactSettings()
    execution: ExecutionSettings = ExecutionSettings()
    retry: RetrySettings = RetrySettings()
    security: SecuritySettings = SecuritySettings()
    modules: dict[str, dict[str, Any]] = {}
    """Raw per-module sections, validated by :meth:`section`."""

    @model_validator(mode="after")
    def _resolve_paths(self) -> FrameworkSettings:
        base = self.base_dir.resolve()

        def absolute(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        updates: dict[str, Any] = {
            "reporting": self.reporting.model_copy(
                update={"output_dir": absolute(self.reporting.output_dir)}
            ),
            "artifacts": self.artifacts.model_copy(update={"root": absolute(self.artifacts.root)}),
        }
        if self.logging.file is not None:
            updates["logging"] = self.logging.model_copy(
                update={"file": absolute(self.logging.file)}
            )
        for name, value in updates.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "base_dir", base)
        return self

    def section(self, name: str, model: type[SectionT]) -> SectionT:
        """Validate ``modules.<name>`` with ``model``; a missing section validates ``{}``."""
        raw = self.modules.get(name, {})
        try:
            return model.model_validate(raw)
        except ValidationError as error:
            raise ConfigurationError(
                f"invalid configuration in modules.{name}:\n{_describe(error, f'modules.{name}')}"
            ) from None

    def redactor(self) -> Redactor:
        return self.security.redactor()

    def redacted_dump(self) -> dict[str, Any]:
        """Settings as JSON-compatible data with sensitive values masked."""
        return self.redactor().redact(self.model_dump(mode="json"))

    def __repr__(self) -> str:
        return f"FrameworkSettings({self.redacted_dump()!r})"

    __str__ = __repr__


def load_settings(
    *,
    config_dir: str | Path | None = None,
    environment: str | None = None,
    env_file: str | Path | bool = True,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> FrameworkSettings:
    """Load and validate framework settings.

    Args:
        config_dir: directory holding ``environments/<env>.yaml``. Defaults to
            ``$CAUTO_CONFIG_DIR``, else ``<project root>/config`` (the nearest
            ``pyproject.toml`` above the working directory), else ``./config``.
        environment: environment name; beats every other source.
        env_file: ``True`` reads ``<base_dir>/.env`` when present, a path reads
            that file (it must exist), ``False`` disables ``.env`` loading.
        environ: process environment to read; defaults to ``os.environ``.
        overrides: nested values applied last.

    Raises:
        ConfigurationError: unreadable files, an unknown explicitly selected
            environment, malformed variables or values that fail validation.
            Messages never include the offending values.
    """
    process_env = dict(os.environ if environ is None else environ)
    overrides = dict(overrides or {})

    directory = _resolve_config_dir(config_dir, process_env)
    base_dir = directory.parent

    dotenv_env = _read_dotenv(env_file, base_dir)
    variables = {**dotenv_env, **process_env}

    selected = (
        environment or overrides.get("environment") or variables.get(f"{ENV_PREFIX}ENVIRONMENT")
    )
    explicit = selected is not None
    env_name = str(selected or DEFAULT_ENVIRONMENT)

    file_values = _read_environment_file(directory, env_name, required=explicit)
    merged: dict[str, Any] = {}
    for layer in (
        file_values,
        _variables_to_tree(dotenv_env, source=".env"),
        _variables_to_tree(process_env, source="environment"),
        overrides,
    ):
        _deep_merge(merged, layer)
    merged["environment"] = env_name
    merged["base_dir"] = base_dir

    try:
        return FrameworkSettings.model_validate(merged)
    except ValidationError as error:
        raise ConfigurationError(
            f"invalid configuration for environment {env_name!r}:\n{_describe(error)}"
        ) from None


def _resolve_config_dir(config_dir: str | Path | None, environ: Mapping[str, str]) -> Path:
    if config_dir is not None:
        directory = Path(config_dir)
    elif environ.get(CONFIG_DIR_VARIABLE):
        directory = Path(environ[CONFIG_DIR_VARIABLE])
    else:
        root = find_project_root()
        directory = (root or Path.cwd()) / "config"
    directory = directory.resolve()
    if directory.exists() and not directory.is_dir():
        raise ConfigurationError(f"configuration path is not a directory: {directory}")
    return directory


def _read_dotenv(env_file: str | Path | bool, base_dir: Path) -> dict[str, str]:
    if env_file is False:
        return {}
    if env_file is True:
        path = base_dir / ".env"
        if not path.is_file():
            return {}
    else:
        path = Path(env_file)
        if not path.is_file():
            raise ConfigurationError(f".env file not found: {path}")
    values = dotenv_values(path)
    return {k: v for k, v in values.items() if k.startswith(ENV_PREFIX) and v is not None}


def _read_environment_file(directory: Path, env_name: str, *, required: bool) -> dict[str, Any]:
    if not _is_valid_env_name(env_name):
        raise ConfigurationError(
            f"invalid environment name {env_name!r}: use lower-case letters, digits, '-' or '_'"
        )
    path = directory / "environments" / f"{env_name}.yaml"
    if not path.is_file():
        if not required:
            return {}
        available = sorted(p.stem for p in (directory / "environments").glob("*.yaml"))
        raise ConfigurationError(
            f"unknown environment {env_name!r}: {path} does not exist "
            f"(available: {', '.join(available) or 'none'})"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigurationError(f"cannot read {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"malformed YAML in {path}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level")
    return data


def _is_valid_env_name(name: str) -> bool:
    return bool(name) and name[0].isalnum() and all(c.isalnum() or c in "-_" for c in name) and name == name.lower()


def _variables_to_tree(variables: Mapping[str, str], *, source: str) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for key, raw in variables.items():
        if not key.startswith(ENV_PREFIX) or key in (CONFIG_DIR_VARIABLE, f"{ENV_PREFIX}ENVIRONMENT"):
            continue
        parts = key[len(ENV_PREFIX):].lower().split("__")
        if not all(parts):
            raise ConfigurationError(f"malformed variable name {key!r} in {source}")
        value: Any = raw
        if raw.lstrip().startswith(("[", "{")):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                raise ConfigurationError(f"{key} in {source} looks like JSON but does not parse") from None
        node = tree
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigurationError(f"{key} in {source} conflicts with another variable")
            node = child
        node[parts[-1]] = value
    return tree


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        elif isinstance(value, Mapping):
            target[key] = {}
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _describe(error: ValidationError, prefix: str = "") -> str:
    """List validation problems by location and message, never echoing input values."""
    lines = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        if prefix:
            location = f"{prefix}.{location}" if location else prefix
        lines.append(f"  - {location or '<root>'}: {item['msg']}")
    return "\n".join(lines)
