"""Schema expectation comparison and migration-diff logic on declarative models (no database needed)."""

import pytest
import yaml
from pydantic import ValidationError
from sqlalchemy import types as sat

from complete_automation.database.errors import DataAssertionError, SchemaValidationError
from complete_automation.database.validations.migrations import assert_counts_preserved, assert_schema_changes, diff_snapshots
from complete_automation.database.validations.schema import (
    ColumnExpectation as Col,
)
from complete_automation.database.validations.schema import (
    ForeignKeyExpectation,
    IndexExpectation,
    SchemaExpectation,
    TableExpectation,
    compare_schema,
    type_family,
)

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]


def _actual() -> SchemaExpectation:
    return SchemaExpectation(
        tables=(
            TableExpectation(
                name="users",
                exact_columns=True,
                columns=(
                    Col(name="id", type="integer", nullable=False, primary_key=True, native_type="INTEGER"),
                    Col(name="email", type="string", nullable=False, max_length=255, native_type="VARCHAR(255)", default=None),
                    Col(name="status", type="string", nullable=False, has_default=True, default="active"),
                ),
                primary_key=("id",),
                unique_constraints=(("email",),),
                indexes=(IndexExpectation(columns=("email",), unique=True, name="ix_email"),),
            ),
            TableExpectation(
                name="orders",
                columns=(Col(name="id", type="integer"), Col(name="user_id", type="integer", nullable=False)),
                primary_key=("id",),
                foreign_keys=(ForeignKeyExpectation(columns=("user_id",), referred_table="users", referred_columns=("id",), on_delete="CASCADE"),),
            ),
        ),
        views=("active_users",),
    )


def _kinds(expected: SchemaExpectation) -> set[str]:
    return {f"{d.kind}:{d.subject}" for d in compare_schema(expected, _actual())}


def test_a_matching_expectation_has_no_differences() -> None:
    expected = SchemaExpectation(
        tables=(
            TableExpectation(name="users", columns=(Col(name="email", type="string", nullable=False, max_length=255),), primary_key=("id",)),
            TableExpectation(
                name="orders",
                foreign_keys=(ForeignKeyExpectation(columns=("user_id",), referred_table="users", referred_columns=("id",), on_delete="cascade"),),
            ),
        ),
        views=("active_users",),
    )

    assert compare_schema(expected, _actual()) == []


def test_table_names_alone_do_not_make_schemas_equal() -> None:
    expected = SchemaExpectation(
        tables=(TableExpectation(name="users", columns=(Col(name="email", type="integer", nullable=True, max_length=10),)),)
    )

    assert _kinds(expected) == {
        "type_mismatch:users.email",
        "nullable_mismatch:users.email",
        "max_length_mismatch:users.email",
    }


def test_missing_objects_are_reported() -> None:
    expected = SchemaExpectation(
        tables=(
            TableExpectation(
                name="users",
                columns=(Col(name="nickname", type="string"),),
                unique_constraints=(("nickname",),),
                indexes=(IndexExpectation(columns=("nickname",)),),
                foreign_keys=(ForeignKeyExpectation(columns=("a",), referred_table="b", referred_columns=("c",)),),
                primary_key=("id", "email"),
            ),
            TableExpectation(name="payments"),
        ),
        views=("ghost_view",),
    )

    assert _kinds(expected) == {
        "missing_column:users.nickname",
        "missing_unique_constraint:users(nickname)",
        "missing_index:users(nickname)",
        "missing_foreign_key:users(a) -> b(c)",
        "primary_key_mismatch:users",
        "missing_table:payments",
        "missing_view:ghost_view",
    }


def test_exact_modes_report_extras() -> None:
    expected = SchemaExpectation(
        tables=(TableExpectation(name="users", exact_columns=True, columns=(Col(name="id"),)),), exact_tables=True
    )

    assert _kinds(expected) == {
        "unexpected_column:users.email",
        "unexpected_column:users.status",
        "unexpected_table:orders",
    }


def test_foreign_key_actions_and_index_uniqueness_are_compared() -> None:
    expected = SchemaExpectation(
        tables=(
            TableExpectation(
                name="orders",
                foreign_keys=(ForeignKeyExpectation(columns=("user_id",), referred_table="users", referred_columns=("id",), on_delete="RESTRICT"),),
            ),
            TableExpectation(name="users", indexes=(IndexExpectation(columns=("email",), unique=False),)),
        )
    )

    diffs = {(d.kind, d.expected, d.actual) for d in compare_schema(expected, _actual())}

    assert ("foreign_key_on_delete_mismatch", "RESTRICT", "CASCADE") in diffs
    assert any(kind == "index_uniqueness_mismatch" for kind, *_ in diffs)


def test_defaults_are_compared_after_normalization() -> None:
    from complete_automation.database.validations.schema import _normalize_default

    assert _normalize_default("'active'::character varying") == "active"
    assert _normalize_default("(0)") == "0" and _normalize_default(None) is None
    expected = SchemaExpectation(tables=(TableExpectation(name="users", columns=(Col(name="status", default="'ACTIVE'::text"),)),))
    assert compare_schema(expected, _actual()) == []
    wrong = SchemaExpectation(tables=(TableExpectation(name="users", columns=(Col(name="status", default="other"),)),))
    assert _kinds(wrong) == {"default_mismatch:users.status"}


def test_native_types_are_only_compared_when_requested() -> None:
    assert _kinds(SchemaExpectation(tables=(TableExpectation(name="users", columns=(Col(name="id", native_type="integer"),)),))) == set()
    assert _kinds(SchemaExpectation(tables=(TableExpectation(name="users", columns=(Col(name="id", native_type="BIGINT"),)),))) == {
        "native_type_mismatch:users.id"
    }


def test_expectations_load_from_yaml() -> None:
    document = yaml.safe_load(
        """
        schema: app
        tables:
          - name: users
            columns: [{name: id, type: integer, primary_key: true}]
            primary_key: [id]
        views: [active_users]
        """
    )

    expectation = SchemaExpectation.model_validate(document)

    assert expectation.schema_name == "app" and expectation.table("users").primary_key == ("id",)  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        SchemaExpectation.model_validate({"tables": [{"name": "t", "columns": [{"name": "c", "type": "varchar"}]}]})


@pytest.mark.parametrize(
    ("column_type", "family"),
    [
        (sat.Integer(), "integer"),
        (sat.BigInteger(), "integer"),
        (sat.String(10), "string"),
        (sat.Text(), "string"),
        (sat.Numeric(10, 2), "numeric"),
        (sat.Float(), "float"),
        (sat.Boolean(), "boolean"),
        (sat.DateTime(), "datetime"),
        (sat.TIMESTAMP(), "datetime"),
        (sat.Date(), "date"),
        (sat.Time(), "time"),
        (sat.LargeBinary(), "binary"),
        (sat.JSON(), "json"),
        (sat.Uuid(), "uuid"),
        (sat.NullType(), "other"),
    ],
)
def test_types_map_to_engine_independent_families(column_type: object, family: str) -> None:
    assert type_family(column_type) == family


# --- migration diffs -----------------------------------------------------------------------------


def _before() -> SchemaExpectation:
    return SchemaExpectation(
        tables=(
            TableExpectation(name="users", columns=(Col(name="id", type="integer"), Col(name="name", type="string", max_length=50))),
            TableExpectation(name="legacy", columns=(Col(name="id", type="integer"),)),
        )
    )


def _after() -> SchemaExpectation:
    return SchemaExpectation(
        tables=(
            TableExpectation(
                name="users",
                columns=(Col(name="id", type="integer"), Col(name="name", type="string", max_length=100), Col(name="email", type="string")),
            ),
            TableExpectation(name="audit", columns=(Col(name="id", type="integer"),)),
        )
    )


def test_snapshots_diff_into_added_removed_and_changed() -> None:
    changes = diff_snapshots(_before(), _after())

    assert changes.added_tables == ("audit",) and changes.removed_tables == ("legacy",)
    assert changes.added_columns == {"users": ("email",)} and changes.changed_columns == {"users": ("name",)}
    assert not changes.empty and diff_snapshots(_before(), _before()).empty


def test_a_migration_must_change_exactly_what_it_declares() -> None:
    declared = {
        "added_tables": ["audit"],
        "removed_tables": ["legacy"],
        "added_columns": {"users": ["email"]},
        "changed_columns": {"users": ["name"]},
    }
    assert assert_schema_changes(_before(), _after(), **declared).added_tables == ("audit",)  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError, match="removed_tables"):
        assert_schema_changes(_before(), _after(), added_tables=["audit"], added_columns={"users": ["email"]}, changed_columns={"users": ["name"]})
    with pytest.raises(SchemaValidationError, match="added_columns"):
        assert_schema_changes(_before(), _after(), **{**declared, "added_columns": {}})  # type: ignore[arg-type]


def test_row_counts_must_be_preserved_except_for_declared_deltas() -> None:
    assert_counts_preserved({"a": 3, "b": 5}, {"a": 3, "b": 5})
    assert_counts_preserved({"a": 3}, {"a": 4}, allowed_changes={"a": 1})
    with pytest.raises(DataAssertionError, match="a: expected 3, found 2"):
        assert_counts_preserved({"a": 3}, {"a": 2})
    with pytest.raises(DataAssertionError, match="found None"):
        assert_counts_preserved({"a": 3}, {})
