"""Runnable examples of the database module, executed as tests against temporary SQLite databases."""

try:
    import sqlalchemy  # noqa: F401
except ImportError:  # the "database" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.database.pytest_plugin import *  # noqa: F401,F403
