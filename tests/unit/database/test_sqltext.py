"""The SQL scanner and statement classifier: literals and comments never change a verdict."""

import pytest

from complete_automation.database.sqltext import (
    OperationKind,
    classify,
    fingerprint,
    keywords,
    positional_to_named,
    statement_count,
)

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]

R, W, D, X, A, C = (
    OperationKind.READ,
    OperationKind.WRITE,
    OperationKind.DDL,
    OperationKind.DESTRUCTIVE,
    OperationKind.ADMIN,
    OperationKind.CONTROL,
)


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("SELECT * FROM t", R),
        ("  select 1", R),
        ("VALUES (1)", R),
        ("WITH x AS (SELECT 1) SELECT * FROM x", R),
        ("EXPLAIN SELECT * FROM t", R),
        ("PRAGMA table_info(t)", R),
        ("INSERT INTO t VALUES (1)", W),
        ("UPDATE t SET a = 1 WHERE id = 2", W),
        ("DELETE FROM t WHERE id = 1", W),
        ("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x", W),
        ("CREATE TABLE t (a int)", D),
        ("ALTER TABLE t ADD COLUMN b int", D),
        ("SELECT * INTO copy FROM t", D),
        ("DROP TABLE t", X),
        ("TRUNCATE t", X),
        ("DELETE FROM t", X),
        ("ALTER TABLE t DROP COLUMN b", X),
        ("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", X),
        ("PRAGMA writable_schema = ON", A),
        ("GRANT ALL ON t TO u", A),
        ("COPY t TO PROGRAM 'x'", A),
        ("something odd", A),
        ("COMMIT", C),
        ("BEGIN", C),
        ("SET search_path = other", C),
        ("SAVEPOINT s", C),
    ],
)
def test_classification(sql: str, kind: OperationKind) -> None:
    assert classify(sql).kind is kind


def test_explain_analyze_is_classified_by_the_statement_it_runs() -> None:
    assert classify("EXPLAIN ANALYZE SELECT 1").kind is R
    assert classify("EXPLAIN ANALYZE DELETE FROM t").kind is X
    assert classify("EXPLAIN (ANALYZE) INSERT INTO t VALUES (1)").kind is W


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'DROP TABLE t; DELETE FROM t' AS note",
        "SELECT 1 -- DROP TABLE t",
        "SELECT /* DELETE FROM t */ 1",
        'SELECT "DROP" FROM t',
        "SELECT $$ DROP TABLE t; $$",
        "SELECT $tag$ ; DELETE FROM t $tag$",
        "SELECT E'it\\'s; DROP TABLE t'",
        "SELECT 'it''s; DROP TABLE t'",
    ],
)
def test_keywords_inside_literals_and_comments_are_ignored(sql: str) -> None:
    info = classify(sql)

    assert info.kind is R and info.statements == 1
    assert "DROP" not in keywords(sql) and "DELETE" not in keywords(sql)


def test_a_second_statement_is_counted_even_after_a_literal() -> None:
    assert classify("SELECT 'x'; DROP TABLE t").statements == 2
    assert classify("SELECT 1 /* c */ ; /* d */").statements == 1
    assert statement_count("") == 0 and statement_count(" ; ; ") == 0
    assert statement_count("a; b;") == 2


def test_fingerprint_removes_literals_and_comments_but_keeps_shape() -> None:
    sql = "SELECT * FROM users -- note\nWHERE email = 'a@b.test' AND age > 30 AND id = :id"

    assert fingerprint(sql) == "SELECT * FROM users WHERE email = ? AND age > ? AND id = :id"
    assert "secret" not in fingerprint("INSERT INTO t VALUES ('secret', 42)")


def test_fingerprint_is_truncated() -> None:
    assert len(fingerprint("SELECT " + ", ".join(["col"] * 500), limit=100)) == 100


def test_positional_placeholders_become_named_binds_outside_literals() -> None:
    sql, params = positional_to_named("SELECT * FROM t WHERE a = ? AND b = '?' AND c = ? -- ?", [1, 2])

    assert sql == "SELECT * FROM t WHERE a = :p0 AND b = '?' AND c = :p1 -- ?"
    assert params == {"p0": 1, "p1": 2}


def test_placeholder_count_must_match_the_values() -> None:
    with pytest.raises(ValueError, match="2 positional placeholder"):
        positional_to_named("SELECT ? , ?", [1])
    with pytest.raises(ValueError, match="1 value"):
        positional_to_named("SELECT 1", [1])
