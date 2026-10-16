import pytest
from pydantic import BaseModel, SecretStr

from complete_automation.core.redaction import Redactor

redactor = Redactor()


def test_nested_structures_are_masked_by_key() -> None:
    data = {
        "user": "bob",
        "password": "hunter2",
        "nested": {"api_key": "k-123", "items": [{"access_token": "t-456"}, {"note": "fine"}]},
        "Authorization": "Bearer abc.def",
        "client-secret": "cs-789",
        "headers": ({"X-API-Key": "hk-1"},),
    }

    result = redactor.redact(data)

    assert result["user"] == "bob"
    assert result["password"] == "***"
    assert result["nested"]["api_key"] == "***"
    assert result["nested"]["items"][0]["access_token"] == "***"
    assert result["nested"]["items"][1]["note"] == "fine"
    assert result["Authorization"] == "***"
    assert result["client-secret"] == "***"
    assert result["headers"][0]["X-API-Key"] == "***"


@pytest.mark.parametrize(
    "key", ["prompt_tokens", "max_tokens", "author", "tokenizer", "passed", "keyword", "primary_key_name"]
)
def test_look_alike_keys_are_not_masked(key: str) -> None:
    assert redactor.redact({key: 42}) == {key: 42}


def test_none_under_a_sensitive_key_stays_none() -> None:
    assert redactor.redact({"password": None}) == {"password": None}


def test_input_is_not_mutated() -> None:
    data = {"password": "hunter2", "list": [{"token": "t"}]}

    redactor.redact(data)

    assert data == {"password": "hunter2", "list": [{"token": "t"}]}


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("connect to postgresql://app:pg-pass-1@db:5432/main", "pg-pass-1"),
        ("Authorization: Bearer eyJhbGciOi.abc.def", "eyJhbGciOi.abc.def"),
        ("authorization=Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("sending bearer tok_live_999 now", "tok_live_999"),
        ("password=hunter2&user=bob", "hunter2"),
        ("api_key: sk-abc-123", "sk-abc-123"),
        ('{"client_secret": "cs value with spaces"}', "cs value with spaces"),
        ("Server=db;User Id=sa;Password=Pa55word;", "Pa55word"),
        ("secret_key='q1w2e3'", "q1w2e3"),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
            "MIIEow",
        ),
    ],
)
def test_text_patterns_are_masked(text: str, secret: str) -> None:
    result = redactor.redact_text(text)

    assert secret not in result
    assert "***" in result


def test_url_redaction_keeps_user_and_host() -> None:
    assert (
        redactor.redact_text("https://bob:pw@example.test/path")
        == "https://bob:***@example.test/path"
    )


def test_plain_text_is_unchanged() -> None:
    text = "prompt_tokens: 120, tokenizer=bpe, status=passed"

    assert redactor.redact_text(text) == text


def test_extra_keys_and_custom_mask() -> None:
    custom = Redactor(extra_keys=["Session-ID"], mask="[hidden]")

    assert custom.redact({"session_id": "s-1", "password": "p"}) == {
        "session_id": "[hidden]",
        "password": "[hidden]",
    }
    assert Redactor().redact({"session_id": "s-1"}) == {"session_id": "s-1"}


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(ValueError):
        Redactor(mask="")


def test_secret_types_and_models_are_masked() -> None:
    class Credentials(BaseModel):
        user: str
        token: SecretStr

    result = redactor.redact({"value": SecretStr("s1"), "model": Credentials(user="u", token="t1")})

    assert result == {"value": "***", "model": {"user": "u", "token": "***"}}


def test_cycles_do_not_recurse_forever() -> None:
    data: dict = {"name": "loop"}
    data["self"] = data

    assert redactor.redact(data)["self"] == "<cycle>"
