"""Operation policy: which kinds of statement this connection may run.

Layers, from strongest to weakest:

1. **The database account.** Use least-privilege credentials; nothing here can
   replace that.
2. **Engine-level read-only mode**, set by the adapter on every connection
   (``mode=ro`` and ``query_only`` for SQLite, ``default_transaction_read_only``
   for PostgreSQL). The server itself refuses writes.
3. **This policy**, which classifies each statement (see :mod:`sqltext`) and
   refuses what the settings do not authorize, with a message that says why.
   Statement classification is a keyword-level guardrail against mistakes. It is
   not, and must not be treated as, a defense against hostile SQL.

Defaults are conservative: read-only, no DDL, no destructive statements. DDL and
destructive statements additionally need the *server-reported* database name to
be in ``privileged_databases`` (so a mistyped host cannot redirect a ``DROP``
at another database), and are impossible in a protected environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from complete_automation.database.errors import DatabaseConfigurationError, UnsafeOperationError
from complete_automation.database.settings import DatabaseSettings
from complete_automation.database.sqltext import OperationKind, StatementInfo

_PRIVILEGED = (OperationKind.DDL, OperationKind.ADMIN, OperationKind.DESTRUCTIVE)


def database_matches(name: str, allowed: tuple[str, ...]) -> bool:
    """True when ``name`` equals an allowlist entry, or its file name does (SQLite paths)."""
    if not name:
        return False
    base = PurePath(name).name
    return any(entry and (entry == name or entry == base) for entry in allowed)


@dataclass(frozen=True)
class OperationPolicy:
    environment: str
    protected: bool
    writable: bool
    allow_ddl: bool
    allow_destructive: bool
    privileged_databases: tuple[str, ...]
    configured_database: str

    @classmethod
    def from_settings(
        cls, settings: DatabaseSettings, *, environment: str, configured_database: str
    ) -> OperationPolicy:
        protected = settings.is_protected(environment)
        wants_more = settings.access == "read_write" or settings.allow_ddl or settings.allow_destructive
        if protected and wants_more:
            raise DatabaseConfigurationError(
                f"environment {environment!r} is protected: modules.database must stay "
                "access: read_only with allow_ddl and allow_destructive off"
            )
        return cls(
            environment=environment,
            protected=protected,
            writable=settings.access == "read_write" and not protected,
            allow_ddl=settings.allow_ddl and not protected,
            allow_destructive=settings.allow_destructive and not protected,
            privileged_databases=settings.privileged_databases,
            configured_database=configured_database,
        )

    @property
    def read_only(self) -> bool:
        return not self.writable

    @staticmethod
    def needs_identity_check(info: StatementInfo) -> bool:
        return info.kind in _PRIVILEGED

    def check(self, info: StatementInfo, *, verified_database: str | None = None) -> None:
        """Raise :class:`UnsafeOperationError` unless ``info`` is allowed here."""
        kind, verb = info.kind, info.verb or "statement"
        suffix = f" ({info.reason})" if info.reason else ""
        if info.statements > 1:
            raise UnsafeOperationError(
                f"refused: {info.statements} statements in one call; run them one at a time"
            )
        if kind is OperationKind.READ:
            return
        if kind is OperationKind.CONTROL:
            raise UnsafeOperationError(
                f"refused: {verb} changes transaction or session state; use the transaction API "
                "(commit/rollback/savepoint) instead of raw SQL"
            )
        if self.protected:
            raise UnsafeOperationError(
                f"refused: {verb}{suffix} is a {kind.value} operation and environment "
                f"{self.environment!r} is protected (read-only)"
            )
        if not self.writable:
            raise UnsafeOperationError(
                f"refused: {verb}{suffix} is a {kind.value} operation but access is read_only "
                "(set modules.database.access: read_write to allow writes)"
            )
        if kind is OperationKind.WRITE:
            return
        if kind is OperationKind.DESTRUCTIVE:
            if not self.allow_destructive:
                raise UnsafeOperationError(
                    f"refused: {verb}{suffix} is destructive; it needs allow_destructive and the "
                    "database in privileged_databases"
                )
        elif not self.allow_ddl:
            raise UnsafeOperationError(
                f"refused: {verb}{suffix} is a {kind.value} operation; it needs allow_ddl and the "
                "database in privileged_databases"
            )
        self._check_identity(verb, verified_database)

    def _check_identity(self, verb: str, verified_database: str | None) -> None:
        if verified_database is None:
            raise UnsafeOperationError(f"refused: could not verify the database identity before {verb}")
        if not database_matches(verified_database, self.privileged_databases):
            raise UnsafeOperationError(
                f"refused: {verb} on database {PurePath(verified_database).name or verified_database!r}, "
                "which is not in privileged_databases"
            )
        if not database_matches(verified_database, (self.configured_database,)):
            raise UnsafeOperationError(
                f"refused: the server reports a different database than configured "
                f"({PurePath(verified_database).name or verified_database!r} vs "
                f"{PurePath(self.configured_database).name or self.configured_database!r})"
            )
