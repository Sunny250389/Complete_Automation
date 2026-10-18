"""UI unit tests: no browser is launched. Need the "ui" extra (Playwright's Python package)."""

try:
    import playwright  # noqa: F401
except ImportError:  # the "ui" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
