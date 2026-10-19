"""ML framework tests with real scikit-learn models, files and the pytest plugin. Need the "ml" extra."""

try:
    import numpy, pandas, sklearn  # noqa: E401,F401
except ImportError:  # the "ml" extra is not installed: leave these tests out
    collect_ignore_glob = ["*.py"]
else:
    from complete_automation.ml.pytest_plugin import *  # noqa: F401,F403
