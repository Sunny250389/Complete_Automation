"""Settings for the UI module, read from ``modules.ui`` of the shared configuration.

Example (``config/environments/<env>.yaml``)::

    modules:
      ui:
        base_url: http://127.0.0.1:3000
        browser: chromium          # chromium | firefox | webkit
        headless: true
        viewport: {width: 1280, height: 720}
        screenshot: on_failure     # never | on_failure | always
        trace: off                 # off | on_failure | always
        video: off                 # off | on_failure | always

Every value can be overridden with ``CAUTO_MODULES__UI__<NAME>`` (for example
``CAUTO_MODULES__UI__BROWSER=firefox``). Credentials for an application under
test are never settings: read them from the environment in the test.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

BrowserName = Literal["chromium", "firefox", "webkit"]
CapturePolicy = Literal["never", "on_failure", "always"]
ToggledCapture = Literal["off", "on_failure", "always"]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Viewport(_Section):
    width: int = Field(default=1280, ge=200, le=7680)
    height: int = Field(default=720, ge=200, le=4320)


class UiSettings(_Section):
    base_url: str | None = None
    browser: BrowserName = "chromium"
    headless: bool = True
    navigation_timeout_seconds: float = Field(default=30.0, gt=0)
    action_timeout_seconds: float = Field(default=10.0, gt=0)
    expect_timeout_seconds: float = Field(default=5.0, gt=0)
    """Timeout of web-first assertions (``expect``)."""
    viewport: Viewport = Viewport()
    locale: str | None = Field(default=None, pattern=r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})*$")
    timezone_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_+\-/]{1,64}$")
    test_id_attribute: str = Field(default="data-testid", pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    launch_args: tuple[str, ...] = ()
    """Extra browser command-line arguments; add only what an environment needs."""
    slow_mo_ms: int = Field(default=0, ge=0, le=5000)
    screenshot: CapturePolicy = "on_failure"
    trace: ToggledCapture = "off"
    """Traces contain network traffic and page contents; they cannot be redacted."""
    video: ToggledCapture = "off"
    max_screenshots_per_test: int = Field(default=3, ge=1, le=20)

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parts = urlsplit(value)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ValueError("base_url must start with http:// or https:// and name a host")
        if parts.username or parts.password:
            raise ValueError("base_url must not contain credentials")
        return value

    @field_validator("launch_args")
    @classmethod
    def _args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for arg in value:
            if not re.fullmatch(r"--?[A-Za-z0-9][^\s]*", arg):
                raise ValueError("launch_args entries must be single flags such as --mute-audio")
        return value

    @property
    def navigation_timeout_ms(self) -> float:
        return self.navigation_timeout_seconds * 1000

    @property
    def action_timeout_ms(self) -> float:
        return self.action_timeout_seconds * 1000

    @property
    def expect_timeout_ms(self) -> float:
        return self.expect_timeout_seconds * 1000
