"""Top-level pytest configuration: shared-core fixtures and pytester for plugin tests."""

from importlib.util import find_spec

pytest_plugins = ["complete_automation.core.pytest_plugin", "pytester"]

# The UI plugin owns session-wide browser fixtures, so it is registered exactly once, here;
# loading it from several conftest.py files would start one Playwright runtime per folder.
if find_spec("playwright") is not None:
    pytest_plugins.append("complete_automation.ui.pytest_plugin")
