import asyncio
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from complete_automation.core.context import (
    RunContext,
    TestContext,
    activate_run_context,
    current_run_context,
    current_test_context,
    deactivate_run_context,
    new_run_id,
    use_test_context,
)


def make_test_context(test_id: str = "tests/a.py::t") -> TestContext:
    return TestContext(run_id="run-1", test_id=test_id, name="t", module="unit")


def test_run_ids_are_sortable_and_unique() -> None:
    ids = {new_run_id() for _ in range(200)}

    assert len(ids) == 200
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", run_id) for run_id in ids)


def test_run_context_start_and_finish() -> None:
    run = RunContext.start("test", suite="smoke", metadata={"branch": "main"})
    finished = run.finish()

    assert run.environment == "test" and run.suite == "smoke"
    assert run.finished_at is None
    assert finished.finished_at is not None and finished.finished_at >= run.started_at
    assert finished.run_id == run.run_id


def test_explicit_run_id_is_kept() -> None:
    assert RunContext.start("dev", run_id="shared-run").run_id == "shared-run"


def test_contexts_are_immutable_and_copy_their_metadata() -> None:
    source = {"k": "v"}
    run = RunContext.start("dev", metadata=source)
    source["k"] = "changed"

    assert run.metadata == {"k": "v"}
    with pytest.raises(ValidationError):
        run.environment = "prod"  # type: ignore[misc]


def test_contexts_serialize_to_json() -> None:
    data = make_test_context().model_dump(mode="json")

    json.dumps(data)
    assert data["started_at"].endswith("Z")  # UTC, ISO 8601
    assert len(data["correlation_id"]) == 16


def test_metadata_must_be_json_compatible() -> None:
    with pytest.raises(ValidationError):
        TestContext(run_id="r", test_id="t", name="t", module="m", metadata={"bad": object()})


def test_each_test_context_gets_its_own_correlation_id() -> None:
    assert make_test_context().correlation_id != make_test_context().correlation_id


def test_use_test_context_resets_even_when_the_body_fails() -> None:
    with pytest.raises(RuntimeError), use_test_context(make_test_context()):
        raise RuntimeError("test failed")

    assert current_test_context() is None


def test_threads_do_not_see_each_others_test_context() -> None:
    barrier = threading.Barrier(8)

    def worker(index: int) -> str:
        with use_test_context(make_test_context(f"tests/a.py::t{index}")):
            barrier.wait()  # every thread holds a context at the same time
            return current_test_context().test_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        seen = list(pool.map(worker, range(8)))

    assert seen == [f"tests/a.py::t{i}" for i in range(8)]
    assert current_test_context() is None


def test_asyncio_tasks_do_not_see_each_others_test_context() -> None:
    async def task(index: int) -> str:
        with use_test_context(make_test_context(f"tests/a.py::t{index}")):
            await asyncio.sleep(0)
            return current_test_context().test_id

    async def main() -> list[str]:
        return await asyncio.gather(*(task(i) for i in range(5)))

    assert asyncio.run(main()) == [f"tests/a.py::t{i}" for i in range(5)]


def test_run_context_activation_pair() -> None:
    previous = current_run_context()
    run = RunContext.start("dev", run_id="activated")

    token = activate_run_context(run)
    assert current_run_context() is run
    deactivate_run_context(token)

    assert current_run_context() is previous
