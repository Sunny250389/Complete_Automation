"""Database adapters. Add an engine by subclassing :class:`DatabaseAdapter` and registering it."""

from complete_automation.database.adapters.base import Capabilities, DatabaseAdapter
from complete_automation.database.adapters.postgresql import PostgreSqlAdapter
from complete_automation.database.adapters.sqlite import SqliteAdapter
from complete_automation.database.errors import DatabaseConfigurationError

_ADAPTERS: dict[str, type[DatabaseAdapter]] = {
    SqliteAdapter.dialect: SqliteAdapter,
    PostgreSqlAdapter.dialect: PostgreSqlAdapter,
}


def register_adapter(adapter: type[DatabaseAdapter]) -> None:
    """Make ``adapter`` available under its ``dialect``; refuses to replace an existing one."""
    if adapter.dialect in _ADAPTERS and _ADAPTERS[adapter.dialect] is not adapter:
        raise DatabaseConfigurationError(f"an adapter for {adapter.dialect!r} is already registered")
    _ADAPTERS[adapter.dialect] = adapter


def get_adapter_class(dialect: str) -> type[DatabaseAdapter]:
    try:
        return _ADAPTERS[dialect]
    except KeyError:
        available = ", ".join(sorted(_ADAPTERS))
        raise DatabaseConfigurationError(f"no adapter for dialect {dialect!r} (available: {available})") from None


__all__ = [
    "Capabilities",
    "DatabaseAdapter",
    "PostgreSqlAdapter",
    "SqliteAdapter",
    "get_adapter_class",
    "register_adapter",
]
