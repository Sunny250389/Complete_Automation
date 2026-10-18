"""DatabaseSettings validation, URL redaction and the operation policy."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from complete_automation.core.config import load_settings
from complete_automation.core.redaction import Redactor
from complete_automation.database.errors import DatabaseConfigurationError, UnsafeOperationError
from complete_automation.database.safety import OperationPolicy, database_matches
from complete_automation.database.settings import DatabaseSettings, safe_url
from complete_automation.database.sqltext import classify

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]


def test_defaults_are_conservative() -> None:
    settings = DatabaseSettings()

    assert settings.access == "read_only" and not settings.allow_ddl and not settings.allow_destructive
    assert settings.privileged_databases == () and not settings.is_configured
    assert settings.capture_diagnostics == "on_failure"
    assert settings.is_protected("prod") and settings.is_protected("Production") and not settings.is_protected("dev")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"url": "sqlite:///x.db", "host": "h"}, "url cannot be combined"),
        ({"url": "sqlite:///x.db", "database": "d"}, "url cannot be combined"),
        ({"url": "not a url"}, "valid SQLAlchemy URL"),
        ({"allow_ddl": True, "access": "read_write"}, "privileged_databases"),
        ({"allow_destructive": True, "access": "read_write"}, "privileged_databases"),
        ({"allow_ddl": True, "privileged_databases": ("d",)}, "access: read_write"),
        ({"schema": "bad-name;drop"}, "plain identifier"),
        ({"port": 70000}, "less than or equal"),
        ({"host": "  "}, "must not be blank"),
        ({"unknown_option": 1}, "Extra inputs"),
    ],
)
def test_invalid_settings_are_rejected(values: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DatabaseSettings(**values)


def test_a_bad_url_error_does_not_echo_the_url() -> None:
    with pytest.raises(ValidationError) as error:
        DatabaseSettings(url="postgresql+psycopg://user:hunter2@host:notaport/db")  # type: ignore[arg-type]

    assert "hunter2" not in str(error.value)


def test_secrets_never_appear_in_repr_or_dump() -> None:
    settings = DatabaseSettings(dialect="postgresql", host="h", database="d", username="u", password="hunter2")  # type: ignore[arg-type]
    url_settings = DatabaseSettings(url="postgresql+psycopg://u:hunter3@h/d")  # type: ignore[arg-type]

    for text in (repr(settings), str(settings), repr(settings.model_dump()), repr(url_settings), str(url_settings.model_dump())):
        assert "hunter2" not in text and "hunter3" not in text


def test_safe_url_hides_passwords_and_sensitive_query_values() -> None:
    url = make_url("postgresql+psycopg://user:hunter2@db.example.test:5432/app?sslmode=require&password=hunter4&token=t0k")

    rendered = safe_url(url, Redactor())

    assert "hunter2" not in rendered and "hunter4" not in rendered and "t0k" not in rendered
    assert "db.example.test" in rendered and "sslmode=require" in rendered and "user" in rendered


def test_dialect_comes_from_the_setting_or_the_url() -> None:
    assert DatabaseSettings(url="postgresql+psycopg://u:p@h/d").resolved_dialect == "postgresql"  # type: ignore[arg-type]
    assert DatabaseSettings(url="sqlite:///x.db").resolved_dialect == "sqlite"  # type: ignore[arg-type]
    assert DatabaseSettings(dialect="sqlite", database=":memory:").resolved_dialect == "sqlite"
    with pytest.raises(DatabaseConfigurationError, match="dialect is not set"):
        _ = DatabaseSettings(database="d").resolved_dialect
    with pytest.raises(DatabaseConfigurationError, match="unsupported database engine"):
        _ = DatabaseSettings(url="mysql+pymysql://u:p@h/d").resolved_dialect  # type: ignore[arg-type]


def test_relative_sqlite_paths_resolve_against_the_base_directory(tmp_path: Path) -> None:
    resolved = DatabaseSettings(dialect="sqlite", database="data/app.db").with_base_dir(tmp_path)

    assert Path(resolved.database or "") == (tmp_path / "data" / "app.db").resolve()
    memory = DatabaseSettings(dialect="sqlite", database=":memory:").with_base_dir(tmp_path)
    assert memory.database == ":memory:"


def test_settings_load_from_the_shared_configuration(tmp_path: Path) -> None:
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "dev.yaml").write_text(
        "modules:\n  database:\n    dialect: sqlite\n    database: local.db\n", encoding="utf-8"
    )
    shared = load_settings(
        config_dir=tmp_path,
        env_file=False,
        environ={
            "CAUTO_MODULES__DATABASE__PRIVILEGED_DATABASES": '["local.db"]',
            "CAUTO_MODULES__DATABASE__ACCESS": "read_write",
        },
    )

    settings = shared.section("database", DatabaseSettings)

    assert settings.dialect == "sqlite" and settings.access == "read_write"
    assert settings.privileged_databases == ("local.db",)


# --- operation policy ------------------------------------------------------------------------------


def _policy(environment: str = "dev", **overrides: Any) -> OperationPolicy:
    values: dict[str, Any] = {"dialect": "sqlite", "database": "app.db", **overrides}
    return OperationPolicy.from_settings(
        DatabaseSettings(**values), environment=environment, configured_database="app.db"
    )


def test_read_only_policy_allows_reads_and_refuses_everything_else() -> None:
    policy = _policy()

    policy.check(classify("SELECT 1"))
    for sql in ("INSERT INTO t VALUES (1)", "UPDATE t SET a=1 WHERE id=1", "CREATE TABLE t(a)", "DROP TABLE t", "PRAGMA x = 1"):
        with pytest.raises(UnsafeOperationError, match="read_only"):
            policy.check(classify(sql))


def test_control_statements_and_multiple_statements_are_always_refused() -> None:
    policy = _policy(access="read_write", allow_ddl=True, allow_destructive=True, privileged_databases=("app.db",))

    with pytest.raises(UnsafeOperationError, match="transaction API"):
        policy.check(classify("COMMIT"))
    with pytest.raises(UnsafeOperationError, match="2 statements"):
        policy.check(classify("SELECT 1; SELECT 2"))


def test_writes_need_read_write_but_not_ddl_rights() -> None:
    policy = _policy(access="read_write")

    policy.check(classify("INSERT INTO t VALUES (1)"))
    policy.check(classify("DELETE FROM t WHERE id = 1"))
    with pytest.raises(UnsafeOperationError, match="allow_ddl"):
        policy.check(classify("CREATE TABLE t(a)"), verified_database="app.db")
    with pytest.raises(UnsafeOperationError, match="destructive"):
        policy.check(classify("DELETE FROM t"), verified_database="app.db")


def test_ddl_and_destructive_statements_need_a_verified_allowlisted_database() -> None:
    policy = _policy(access="read_write", allow_ddl=True, allow_destructive=True, privileged_databases=("app.db",))

    policy.check(classify("CREATE TABLE t(a)"), verified_database="/data/app.db")
    policy.check(classify("DROP TABLE t"), verified_database="app.db")
    with pytest.raises(UnsafeOperationError, match="could not verify"):
        policy.check(classify("DROP TABLE t"))
    with pytest.raises(UnsafeOperationError, match="not in privileged_databases"):
        policy.check(classify("DROP TABLE t"), verified_database="other.db")


def test_a_server_reporting_a_different_database_than_configured_is_refused() -> None:
    settings = DatabaseSettings(
        dialect="postgresql",
        database="app_test",
        access="read_write",
        allow_ddl=True,
        privileged_databases=("app_test", "prod_db"),
    )
    policy = OperationPolicy.from_settings(settings, environment="dev", configured_database="app_test")

    with pytest.raises(UnsafeOperationError, match="different database than configured"):
        policy.check(classify("CREATE TABLE t(a int)"), verified_database="prod_db")


def test_protected_environments_cannot_be_made_writable() -> None:
    read_only = _policy(environment="prod")
    read_only.check(classify("SELECT 1"))
    with pytest.raises(UnsafeOperationError, match="protected"):
        read_only.check(classify("INSERT INTO t VALUES (1)"))

    for extra in ({"access": "read_write"}, {"access": "read_write", "allow_ddl": True, "privileged_databases": ("app.db",)}):
        with pytest.raises(DatabaseConfigurationError, match="protected"):
            _policy(environment="production", **extra)


def test_database_matching_uses_names_not_substrings() -> None:
    assert database_matches("/x/app.db", ("app.db",)) and database_matches("app", ("app",))
    assert not database_matches("/x/myapp.db", ("app.db",)) and not database_matches("app_prod", ("app",))
    assert not database_matches("x", ())
    assert not database_matches("", ("",)) and not database_matches("", ("app.db",))
    with pytest.raises(ValidationError, match="must not be blank"):
        DatabaseSettings(dialect="sqlite", database="a.db", privileged_databases=("",))
