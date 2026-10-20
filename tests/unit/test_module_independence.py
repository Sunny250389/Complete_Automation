"""Each testing module stays usable on its own: importing it never loads another domain module.

The per-module architecture tests check each file's own import statements; this checks
what actually ends up loaded, transitively, in a fresh interpreter, for every package and
its pytest plugin. A circular import between modules would fail here too.
"""

import json
import subprocess
import sys
from importlib.util import find_spec

import pytest

pytestmark = [pytest.mark.cross_module, pytest.mark.cauto_module("cross_module")]

DOMAINS = ("api", "database", "ui", "ml", "llm_evaluation")
REQUIRES = {"api": "httpx", "database": "sqlalchemy", "ui": "playwright", "ml": "sklearn", "llm_evaluation": "httpx"}
TARGETS = [
    *(f"complete_automation.{name}" for name in DOMAINS),
    *(f"complete_automation.{name}.pytest_plugin" for name in DOMAINS),
]

SCRIPT = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
domains = {"api", "database", "ui", "ml", "llm_evaluation"}
print(json.dumps(sorted({name.split(".")[1] for name in sys.modules
                         if name.startswith("complete_automation.") and name.split(".")[1] in domains})))
"""


@pytest.mark.parametrize("target", TARGETS)
def test_importing_a_module_loads_no_other_domain_module(target: str) -> None:
    domain = target.split(".")[1]
    if find_spec(REQUIRES[domain]) is None:
        pytest.skip(f"the {domain} extra is not installed")
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, target], capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(completed.stdout) == [domain]
