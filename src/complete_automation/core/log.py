"""Framework logging on top of the standard ``logging`` package.

Nothing is configured at import time. :func:`configure_logging` attaches
handlers to the ``complete_automation`` logger only (never the root logger),
replaces the handlers it attached earlier instead of adding duplicates, and
keeps propagation on so pytest's capture still sees records.

Every record handled by these handlers gets ``run_id``, ``test_id`` and
``correlation_id`` from the current execution context, plus any fields bound
with :func:`log_context`, and passes through redaction before it is written.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

from complete_automation.core.config import LoggingSettings
from complete_automation.core.context import current_run_context, current_test_context
from complete_automation.core.redaction import DEFAULT_REDACTOR, Redactor

ROOT_LOGGER = "complete_automation"
_HANDLER_MARK = "_cauto_handler"
_REDACTION_FAILED = "[log record withheld: redaction failed]"

_fields: ContextVar[dict[str, Any] | None] = ContextVar("cauto_log_fields", default=None)

# Attributes every LogRecord has; anything else came from ``extra=``.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
_CONTEXT_ATTRS = ("run_id", "test_id", "correlation_id", "context")


def get_logger(name: str = "") -> logging.Logger:
    """Return a logger in the framework namespace (``complete_automation.<name>``)."""
    if not name or name == ROOT_LOGGER:
        return logging.getLogger(ROOT_LOGGER)
    if name.startswith(f"{ROOT_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind structured fields to every framework log record in this context."""
    token = _fields.set({**(_fields.get() or {}), **fields})
    try:
        yield
    finally:
        _fields.reset(token)


class ContextFilter(logging.Filter):
    """Adds execution identifiers and bound fields to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        test = current_test_context()
        run = current_run_context()
        record.run_id = test.run_id if test else (run.run_id if run else "-")
        record.test_id = test.test_id if test else "-"
        record.correlation_id = test.correlation_id if test else "-"
        record.context = dict(_fields.get() or {})
        return True


class RedactingFilter(logging.Filter):
    """Masks secrets in the message, its arguments and structured fields."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self.redactor.redact_text(record.getMessage())
            record.args = None
            for key, value in list(vars(record).items()):
                if key in _STANDARD_ATTRS or key in ("run_id", "test_id", "correlation_id"):
                    continue
                if self.redactor.is_sensitive_key(key):
                    setattr(record, key, self.redactor.mask)
                else:
                    setattr(record, key, self.redactor.redact(value))
        except Exception:  # never fall back to the raw record
            record.msg, record.args = _REDACTION_FAILED, None
            for key in list(vars(record)):
                if key not in _STANDARD_ATTRS:
                    setattr(record, key, None)
        return True


class _UtcFormatter(logging.Formatter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        return datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")

    def format(self, record: logging.LogRecord) -> str:
        try:
            return self.redactor.redact_text(self.render(record))
        except Exception:
            return _REDACTION_FAILED

    def render(self, record: logging.LogRecord) -> str:
        raise NotImplementedError


class TextFormatter(_UtcFormatter):
    def render(self, record: logging.LogRecord) -> str:
        line = (
            f"{self.formatTime(record)} {record.levelname:<8} {record.name} "
            f"[run={getattr(record, 'run_id', '-')} test={getattr(record, 'test_id', '-')} "
            f"corr={getattr(record, 'correlation_id', '-')}] {record.getMessage()}"
        )
        extras = _extras(record)
        if extras:
            line += " " + json.dumps(extras, sort_keys=True, default=str)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(_UtcFormatter):
    def render(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "test_id": getattr(record, "test_id", "-"),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    extras = {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_ATTRS and key not in _CONTEXT_ATTRS
    }
    extras.update(getattr(record, "context", None) or {})
    return extras


def configure_logging(
    settings: LoggingSettings,
    *,
    redactor: Redactor | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the framework logger; safe to call repeatedly.

    Handlers attached by a previous call are removed and closed first, so
    repeated calls never duplicate output.
    """
    redactor = redactor or DEFAULT_REDACTOR
    logger = logging.getLogger(ROOT_LOGGER)
    reset_logging()
    logger.setLevel(settings.level)

    formatter: logging.Formatter = (
        JsonFormatter(redactor) if settings.format == "json" else TextFormatter(redactor)
    )
    handlers: list[logging.Handler] = []
    if settings.console:
        handlers.append(logging.StreamHandler(stream or sys.stderr))
    if settings.file is not None:
        settings.file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.file, encoding="utf-8"))

    for handler in handlers:
        setattr(handler, _HANDLER_MARK, True)
        handler.setFormatter(formatter)
        handler.addFilter(ContextFilter())
        handler.addFilter(RedactingFilter(redactor))
        logger.addHandler(handler)
    return logger


def reset_logging() -> None:
    """Remove and close every handler :func:`configure_logging` attached."""
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()
