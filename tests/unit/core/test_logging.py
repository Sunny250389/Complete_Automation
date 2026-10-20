import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from complete_automation.core.config import LoggingSettings
from complete_automation.core.context import (
    RunContext,
    TestContext,
    current_run_context,
    use_run_context,
    use_test_context,
)
from complete_automation.core.log import (
    ROOT_LOGGER,
    configure_logging,
    get_logger,
    log_context,
    reset_logging,
)
from complete_automation.core.redaction import Redactor

pytestmark = pytest.mark.cauto_module("core")


@pytest.fixture(autouse=True)
def _clean_logging() -> Iterator[None]:
    reset_logging()
    yield
    reset_logging()
    logging.getLogger(ROOT_LOGGER).setLevel(logging.NOTSET)


def framework_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger(ROOT_LOGGER).handlers if getattr(h, "_cauto_handler", False)]


def test_repeated_configuration_does_not_duplicate_handlers() -> None:
    stream = io.StringIO()
    for _ in range(3):
        configure_logging(LoggingSettings(), stream=stream)

    get_logger("dup").info("once")

    assert len(framework_handlers()) == 1
    assert stream.getvalue().count("once") == 1


def test_level_is_applied() -> None:
    stream = io.StringIO()
    configure_logging(LoggingSettings(level="WARNING"), stream=stream)

    get_logger("levels").info("hidden-info")
    get_logger("levels").warning("shown-warning")

    assert "hidden-info" not in stream.getvalue()
    assert "shown-warning" in stream.getvalue()


def test_text_format_carries_execution_identifiers() -> None:
    stream = io.StringIO()
    configure_logging(LoggingSettings(), stream=stream)
    run = RunContext.start("dev", run_id="run-1")
    test = TestContext(run_id="run-1", test_id="tests/x.py::t", name="t", module="unit", correlation_id="c-9")

    with use_run_context(run), use_test_context(test):
        get_logger("ctx").info("inside")
    get_logger("ctx").info("outside")

    inside, outside = stream.getvalue().strip().splitlines()
    assert "run=run-1 test=tests/x.py::t corr=c-9" in inside
    assert "INFO" in inside and "complete_automation.ctx" in inside
    assert inside[:4].isdigit() and "+00:00" in inside.split()[0]
    session_run = current_run_context()  # set by the shared-core plugin for this pytest run
    expected_run = session_run.run_id if session_run else "-"
    assert f"run={expected_run} test=- corr=-" in outside


def test_json_format_includes_structured_fields() -> None:
    stream = io.StringIO()
    configure_logging(LoggingSettings(format="json"), stream=stream)

    with log_context(step="login", attempt=2):
        get_logger("json").info("structured", extra={"endpoint": "/users"})

    record = json.loads(stream.getvalue())
    assert record["message"] == "structured"
    assert record["logger"] == "complete_automation.json"
    assert record["step"] == "login" and record["attempt"] == 2
    assert record["endpoint"] == "/users"
    assert record["test_id"] == "-"


def test_log_context_is_scoped() -> None:
    stream = io.StringIO()
    configure_logging(LoggingSettings(format="json"), stream=stream)

    with log_context(step="a"):
        pass
    get_logger("scope").info("after")

    assert "step" not in json.loads(stream.getvalue())


def test_secrets_are_redacted_from_messages_arguments_and_fields() -> None:
    stream = io.StringIO()
    configure_logging(LoggingSettings(format="json"), stream=stream)

    with log_context(password="ctx-pw-1"):
        get_logger("secret").info(
            "calling with token=%s",
            "arg-tok-2",
            extra={"api_key": "extra-key-3", "request": {"headers": {"Authorization": "Bearer hdr-4"}}},
        )

    output = stream.getvalue()
    for secret in ("ctx-pw-1", "arg-tok-2", "extra-key-3", "hdr-4"):
        assert secret not in output
    assert "***" in output


def test_exception_tracebacks_are_redacted() -> None:
    stream = io.StringIO()
    configure_logging(LoggingSettings(), stream=stream)

    try:
        raise RuntimeError("login failed with password=hunter2")
    except RuntimeError:
        get_logger("exc").exception("boom")

    assert "hunter2" not in stream.getvalue()
    assert "RuntimeError" in stream.getvalue()


def test_failed_redaction_withholds_the_record() -> None:
    class BrokenRedactor(Redactor):
        def redact_text(self, text: str) -> str:
            raise RuntimeError("broken")

    stream = io.StringIO()
    configure_logging(LoggingSettings(), redactor=BrokenRedactor(), stream=stream)

    get_logger("broken").info("password=raw-secret")

    assert "raw-secret" not in stream.getvalue()
    assert "redaction failed" in stream.getvalue()


def test_configured_extra_keys_are_redacted() -> None:
    stream = io.StringIO()
    configure_logging(
        LoggingSettings(format="json"), redactor=Redactor(extra_keys=["pin"]), stream=stream
    )

    get_logger("extra").info("x", extra={"pin": "1234"})

    assert "1234" not in stream.getvalue()


def test_file_logging(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    configure_logging(LoggingSettings(console=False, file=log_file))

    get_logger("file").warning("to file")
    reset_logging()

    assert "to file" in log_file.read_text(encoding="utf-8")


def test_reset_leaves_foreign_handlers_alone() -> None:
    foreign = logging.NullHandler()
    logger = logging.getLogger(ROOT_LOGGER)
    logger.addHandler(foreign)
    try:
        configure_logging(LoggingSettings(), stream=io.StringIO())
        reset_logging()

        assert foreign in logger.handlers
        assert framework_handlers() == []
    finally:
        logger.removeHandler(foreign)


def test_root_logger_is_untouched() -> None:
    before = list(logging.getLogger().handlers)

    configure_logging(LoggingSettings(), stream=io.StringIO())

    assert logging.getLogger().handlers == before


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "complete_automation"),
        ("api.client", "complete_automation.api.client"),
        ("complete_automation.db", "complete_automation.db"),
    ],
)
def test_get_logger_uses_the_framework_namespace(name: str, expected: str) -> None:
    assert get_logger(name).name == expected
