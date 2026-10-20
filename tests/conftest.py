"""Top-level pytest configuration: shared-core fixtures and pytester for plugin tests."""

import os
from importlib.util import find_spec

import pytest

pytest_plugins = ["complete_automation.core.pytest_plugin", "pytester"]

# The UI plugin owns session-wide browser fixtures, so it is registered exactly once, here;
# loading it from several conftest.py files would start one Playwright runtime per folder.
if find_spec("playwright") is not None:
    pytest_plugins.append("complete_automation.ui.pytest_plugin")


@pytest.fixture
def pytester(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """pytester with none of the shell's ``CAUTO_*`` settings.

    A plugin self-test describes its generated project completely; an exported model name,
    base URL or environment (set for live tests, say) must not change what it expects.
    The outer session still reads those settings.
    """
    for name in [name for name in os.environ if name.startswith("CAUTO_")]:
        monkeypatch.delenv(name)
    return pytester
