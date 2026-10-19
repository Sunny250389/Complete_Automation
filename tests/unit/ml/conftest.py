"""ML unit tests: no files, no network. Need the "ml" extra."""

try:
    import numpy, pandas, sklearn  # noqa: E401,F401
except ImportError:  # the "ml" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
