"""UI fixtures, built on the shared-core fixtures.

Register it once, in the top-level ``conftest.py``::

    pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.ui.pytest_plugin"]

Do not star-import it into several ``conftest.py`` files: session fixtures would then
exist once per folder and start one browser runtime each.

Fixtures (none autouse; no browser starts until a test asks for one):

====================  ========  ====================================================
``ui_settings``       session   ``modules.ui`` settings
``ui_session``        module    :class:`BrowserSession`: Playwright runtime + one browser
``ui_base_url``       session   ``modules.ui.base_url``; skips if unset
``ui_context``        function  fresh isolated ``BrowserContext`` with diagnostics
``ui_page``           function  a ``Page`` in that context
====================  ========  ====================================================

Each test gets its own context, so cookies and storage never leak between
tests. The browser is shared by the tests of one module and closed when the module ends:
while Playwright's sync runtime is alive the thread has a running event loop, so a test
that calls ``asyncio.run`` fails if it sits in a module that also uses the browser. After a test the context's diagnostics are stored per
``modules.ui`` (``screenshot``, ``trace``, ``video``) through the shared
``artifact_store`` and attached to the test's ``TestResult``. Override a fixture
by defining one with the same name in a closer ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from playwright.sync_api import BrowserContext, Page

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.config import FrameworkSettings
from complete_automation.core.log import get_logger
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.ui.browser import BrowserSession
from complete_automation.ui.diagnostics import UiDiagnostics
from complete_automation.ui.settings import UiSettings

__all__ = [
    "pytest_runtest_makereport",
    "ui_base_url",
    "ui_context",
    "ui_page",
    "ui_session",
    "ui_settings",
]

_FAILED = pytest.StashKey[bool]()
_FAILURE_TEXT = pytest.StashKey[str]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    report = yield
    if report.when in ("setup", "call") and report.failed:
        item.stash[_FAILED] = True
        item.stash.setdefault(_FAILURE_TEXT, report.longreprtext or "")
    return report


@pytest.fixture(scope="session")
def ui_settings(framework_settings: FrameworkSettings) -> UiSettings:
    return framework_settings.section("ui", UiSettings)


@pytest.fixture(scope="module")
def ui_session(ui_settings: UiSettings) -> Iterator[BrowserSession]:
    session = BrowserSession.start(ui_settings)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def ui_base_url(ui_settings: UiSettings) -> str:
    if not ui_settings.base_url:
        pytest.skip(
            "modules.ui.base_url is not configured "
            "(set it in the environment file or CAUTO_MODULES__UI__BASE_URL)"
        )
    return ui_settings.base_url


@pytest.fixture
def ui_context(
    request: pytest.FixtureRequest,
    ui_session: BrowserSession,
    ui_settings: UiSettings,
    framework_settings: FrameworkSettings,
    artifact_store: LocalArtifactStore,
    result_recorder: ResultRecorder,
    test_context: Any,
) -> Iterator[BrowserContext]:
    diagnostics = UiDiagnostics(
        settings=ui_settings,
        store=artifact_store,
        recorder=result_recorder,
        redactor=framework_settings.redactor(),
        logger=get_logger("ui"),
        test_id=request.node.nodeid,
        test_name=request.node.name,
        browser_version=ui_session.version,
    )
    try:
        context = ui_session.new_context(video_dir=diagnostics.prepare_video_dir())
    except BaseException:
        diagnostics.after_close(failed=True)  # removes the scratch directory
        raise
    diagnostics.watch(context)
    try:
        yield context
    finally:
        failed = request.node.stash.get(_FAILED, False)
        try:
            diagnostics.before_close(
                context, failed=failed, failure_text=request.node.stash.get(_FAILURE_TEXT, None)
            )
        finally:
            try:
                context.close()
            finally:
                diagnostics.after_close(failed=failed)


@pytest.fixture
def ui_page(ui_context: BrowserContext) -> Page:
    return ui_context.new_page()
