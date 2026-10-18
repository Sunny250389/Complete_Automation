"""A small SQL scanner: just enough lexing to classify and summarize statements safely.

It is not a SQL parser. It understands string literals (including doubled quotes,
PostgreSQL ``E'...'`` escapes and ``$tag$...$tag$`` quoting), quoted identifiers
and comments, so keywords inside literals or comments never influence
classification and literals never reach a fingerprint. That is enough for a
*guardrail*; it is not a security boundary. The boundary is the database
connection's own read-only mode and the account's privileges.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+")
_DOLLAR_TAG = re.compile(r"\$[A-Za-z_]*\$")
_NAMED_PARAM = re.compile(r":[A-Za-z_][A-Za-z0-9_]*")


class Token(StrEnum):
    WORD = "word"
    STRING = "string"
    QUOTED_IDENTIFIER = "quoted_identifier"
    NUMBER = "number"
    PUNCT = "punct"
    PARAM = "param"
    COMMENT = "comment"
    SPACE = "space"


def tokenize(sql: str) -> Iterator[tuple[Token, str]]:
    """Yield ``(kind, text)`` pairs covering ``sql`` exactly. Unterminated literals run to the end."""
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            j = i + 1
            while j < n and sql[j].isspace():
                j += 1
            yield Token.SPACE, sql[i:j]
            i = j
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            yield Token.COMMENT, sql[i:j]
            i = j
        elif sql.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if sql.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif sql.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            yield Token.COMMENT, sql[i:j]
            i = j
        elif ch == "'" or (ch in "eE" and sql.startswith("'", i + 1) and not _prev_is_word(sql, i)):
            escapes = ch != "'"
            j = i + (2 if escapes else 1)
            while j < n:
                if escapes and sql[j] == "\\":
                    j += 2
                elif sql[j] == "'":
                    if sql.startswith("''", j):
                        j += 2
                    else:
                        j += 1
                        break
                else:
                    j += 1
            yield Token.STRING, sql[i : min(j, n)]
            i = min(j, n)
        elif ch == '"' or ch == "`":
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if sql.startswith(ch * 2, j):
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            yield Token.QUOTED_IDENTIFIER, sql[i : min(j, n)]
            i = min(j, n)
        elif ch == "$" and (tag := _DOLLAR_TAG.match(sql, i)):
            end = sql.find(tag.group(), tag.end())
            j = n if end < 0 else end + len(tag.group())
            yield Token.STRING, sql[i:j]
            i = j
        elif ch == ":" and not sql.startswith("::", i) and (m := _NAMED_PARAM.match(sql, i)):
            yield Token.PARAM, m.group()
            i = m.end()
        elif ch == ":" and sql.startswith("::", i):
            yield Token.PUNCT, "::"
            i += 2
        elif ch == "?":
            yield Token.PARAM, "?"
            i += 1
        elif (m := _WORD.match(sql, i)):
            yield Token.WORD, m.group()
            i = m.end()
        elif (m := _NUMBER.match(sql, i)):
            yield Token.NUMBER, m.group()
            i = m.end()
        else:
            yield Token.PUNCT, ch
            i += 1


def _prev_is_word(sql: str, index: int) -> bool:
    return index > 0 and (sql[index - 1].isalnum() or sql[index - 1] in "_$")


def keywords(sql: str) -> list[str]:
    """Upper-cased words outside literals, quoted identifiers and comments."""
    return [text.upper() for kind, text in tokenize(sql) if kind is Token.WORD]


def statement_count(sql: str) -> int:
    """Number of non-empty statements separated by ``;`` outside literals and comments."""
    count, has_content = 0, False
    for kind, text in tokenize(sql):
        if kind is Token.PUNCT and text == ";":
            count += has_content
            has_content = False
        elif kind not in (Token.SPACE, Token.COMMENT):
            has_content = True
    return count + has_content


def fingerprint(sql: str, *, limit: int = 300) -> str:
    """The statement with comments dropped and every literal replaced by ``?``.

    Safe to log and report: it keeps the shape (tables, columns, bind names) and
    none of the values that were written inline.
    """
    parts: list[str] = []
    for kind, text in tokenize(sql):
        if kind is Token.COMMENT:
            continue
        if kind in (Token.STRING, Token.NUMBER):
            parts.append("?")
        elif kind is Token.SPACE:
            parts.append(" ")
        else:
            parts.append(text)
    return re.sub(r" +", " ", "".join(parts)).strip()[:limit]


def positional_to_named(sql: str, values: Sequence[object]) -> tuple[str, dict[str, object]]:
    """Turn ``?`` placeholders into named binds ``:p0``, ``:p1``... and zip them with ``values``.

    Only ``?`` outside literals, quoted identifiers and comments is replaced. The
    number of placeholders must equal ``len(values)``.
    """
    names: list[str] = []
    out: list[str] = []
    for kind, text in tokenize(sql):
        if kind is Token.PARAM and text == "?":
            names.append(f"p{len(names)}")
            out.append(f":{names[-1]}")
        else:
            out.append(text)
    if len(names) != len(values):
        raise ValueError(
            f"statement has {len(names)} positional placeholder(s) but {len(values)} value(s) were given"
        )
    return "".join(out), dict(zip(names, values, strict=True))


class OperationKind(StrEnum):
    READ = "read"
    WRITE = "write"
    DDL = "ddl"
    DESTRUCTIVE = "destructive"
    ADMIN = "admin"
    CONTROL = "control"
    """Transaction control or session state (BEGIN, COMMIT, SET ...): use the transaction API."""


@dataclass(frozen=True)
class StatementInfo:
    kind: OperationKind
    verb: str
    """First keyword, upper case (``SELECT``, ``INSERT`` ...); ``""`` for an empty statement."""
    statements: int
    reason: str = ""
    """Why the statement got a stricter kind than its verb suggests."""


_READ_VERBS = frozenset({"SELECT", "VALUES", "SHOW", "TABLE"})
_WRITE_VERBS = frozenset({"INSERT", "UPDATE", "REPLACE", "MERGE", "UPSERT"})
_DDL_VERBS = frozenset({"CREATE", "ALTER", "COMMENT", "REINDEX", "VACUUM", "CLUSTER", "REFRESH"})
_DESTRUCTIVE_VERBS = frozenset({"DROP", "TRUNCATE"})
_CONTROL_VERBS = frozenset(
    {
        "BEGIN", "COMMIT", "ROLLBACK", "END", "ABORT", "SAVEPOINT", "RELEASE", "START", "SET",
        "RESET", "DEALLOCATE", "DISCARD", "USE",
    }
)  # fmt: skip
_DML_WORDS = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE"})


def classify(sql: str) -> StatementInfo:
    """Classify one SQL string conservatively; anything not recognized is ``ADMIN``."""
    words = keywords(sql)
    count = statement_count(sql)
    verb = words[0] if words else ""
    if not words:
        return StatementInfo(OperationKind.READ, "", count)
    if verb in _CONTROL_VERBS:
        return StatementInfo(OperationKind.CONTROL, verb, count)
    if verb in _DESTRUCTIVE_VERBS:
        return StatementInfo(OperationKind.DESTRUCTIVE, verb, count)
    if verb in _DDL_VERBS:
        if "DROP" in words:
            return StatementInfo(OperationKind.DESTRUCTIVE, verb, count, "contains DROP")
        return StatementInfo(OperationKind.DDL, verb, count)
    if verb == "DELETE":
        if "WHERE" not in words:
            return StatementInfo(OperationKind.DESTRUCTIVE, verb, count, "DELETE without WHERE")
        return StatementInfo(OperationKind.WRITE, verb, count)
    if verb in _WRITE_VERBS:
        return StatementInfo(OperationKind.WRITE, verb, count)
    if verb == "EXPLAIN":
        if "ANALYZE" in words or "ANALYSE" in words:  # EXPLAIN ANALYZE executes the statement
            inner = next((i for i, w in enumerate(words[1:], 1) if w in _INNER_VERBS), None)
            if inner is not None:
                info = classify(_after_word(sql, words[inner], inner))
                return StatementInfo(info.kind, verb, count, f"EXPLAIN ANALYZE of {info.verb}")
        return StatementInfo(OperationKind.READ, verb, count)
    if verb in _READ_VERBS:
        if verb == "SELECT" and "INTO" in words:
            return StatementInfo(OperationKind.DDL, verb, count, "SELECT INTO creates a table")
        return StatementInfo(OperationKind.READ, verb, count)
    if verb == "WITH":
        if "DELETE" in words and "WHERE" not in words:
            return StatementInfo(OperationKind.DESTRUCTIVE, verb, count, "CTE DELETE without WHERE")
        if _DML_WORDS & set(words):
            return StatementInfo(OperationKind.WRITE, verb, count, "CTE contains a data change")
        return StatementInfo(OperationKind.READ, verb, count)
    if verb == "PRAGMA":
        if any(kind is Token.PUNCT and text == "=" for kind, text in tokenize(sql)):
            return StatementInfo(OperationKind.ADMIN, verb, count, "PRAGMA assignment")
        return StatementInfo(OperationKind.READ, verb, count)
    return StatementInfo(OperationKind.ADMIN, verb, count, "unrecognized statement type")


_INNER_VERBS = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE", "MERGE", "WITH", "CREATE", "DROP"})


def _after_word(sql: str, word: str, occurrence: int) -> str:
    """The text starting at the ``occurrence``-th (0-based, counting from the first) keyword."""
    seen = -1
    position = 0
    for kind, text in tokenize(sql):
        if kind is Token.WORD:
            seen += 1
            if seen == occurrence and text.upper() == word:
                return sql[position:]
        position += len(text)
    return sql
