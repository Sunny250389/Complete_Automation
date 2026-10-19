"""A deterministic local web site for UI framework self-tests and examples (standard library only).

It serves the static pages next to this file on 127.0.0.1 and a free port, so no
network access or external site is needed. ``/boom.html`` logs a console error
containing a fake credential, to exercise diagnostics and redaction.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from complete_automation.core.config import FrameworkSettings
from complete_automation.ui.settings import UiSettings

SITE_DIR = Path(__file__).resolve().parent


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - signature from the base class
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class RunningSite:
    """The site on ``http://127.0.0.1:<port>``; stop it with :meth:`close`."""

    def __init__(self) -> None:
        handler = partial(_QuietHandler, directory=str(SITE_DIR))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


_lock = threading.Lock()
_site: RunningSite | None = None
_users = 0


def acquire_site() -> RunningSite:
    """The one shared site; started by the first user and stopped when the last one releases it."""
    global _site, _users
    with _lock:
        if _site is None:
            _site = RunningSite()
        _users += 1
        return _site


def release_site() -> None:
    global _site, _users
    with _lock:
        _users -= 1
        if _users == 0 and _site is not None:
            _site.close()
            _site = None


@pytest.fixture(scope="session")
def ui_site() -> Iterator[str]:
    """Base URL of the local practice site (one server per run, however many conftests import this)."""
    site = acquire_site()
    try:
        yield site.url
    finally:
        release_site()


@pytest.fixture(scope="session")
def ui_settings(framework_settings: FrameworkSettings, ui_site: str) -> UiSettings:
    """``modules.ui`` settings (browser, headless, diagnostics ...) pointed at the local site.

    Framework self-tests always target the local site; to test your own application
    use ``ui_settings`` from the plugin unchanged (see docs/ui_automation.md).
    """
    return framework_settings.section("ui", UiSettings).model_copy(update={"base_url": ui_site})
