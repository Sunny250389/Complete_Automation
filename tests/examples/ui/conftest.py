"""Example UI tests against the local practice site. Need the "ui" extra and an installed browser (see docs/ui_automation.md)."""

try:
    import playwright  # noqa: F401
except ImportError:  # the "ui" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from tests.support.ui_site.server import ui_settings, ui_site  # noqa: F401
