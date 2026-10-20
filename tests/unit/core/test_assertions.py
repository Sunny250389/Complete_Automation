import pytest

from complete_automation.core.assertions import assert_keys_present, assert_subset

pytestmark = pytest.mark.cauto_module("core")


def test_keys_present_passes() -> None:
    assert_keys_present({"id": 1, "name": "a", "extra": True}, ["id", "name"])


def test_missing_keys_are_listed() -> None:
    with pytest.raises(AssertionError) as raised:
        assert_keys_present({"id": 1}, ["id", "name", "email"], label="user")

    assert str(raised.value) == "user: missing required key(s) ['email', 'name']; present: ['id']"


def test_keys_present_requires_a_mapping() -> None:
    with pytest.raises(AssertionError, match="expected a mapping, got list"):
        assert_keys_present([1, 2], ["id"])  # type: ignore[arg-type]


def test_subset_allows_extra_keys() -> None:
    assert_subset({"id": 1, "tags": ["a"], "meta": {"x": 1, "y": 2}}, {"tags": ["a"], "meta": {"x": 1}})


def test_subset_reports_every_mismatching_path() -> None:
    actual = {"user": {"name": "bob", "age": 30}, "items": [1, 2], "flag": 1}
    expected = {"user": {"name": "alice", "email": "a@x"}, "items": [1, 2, 3], "flag": True}

    with pytest.raises(AssertionError) as raised:
        assert_subset(actual, expected, label="body")

    message = str(raised.value)
    assert message.startswith("body: 4 mismatch(es):")
    assert "body.user.name: expected 'alice', got 'bob'" in message
    assert "body.user.email: missing" in message
    assert "body.items: expected 3 item(s), got 2" in message
    assert "body.flag: expected True, got 1" in message  # 1 == True, but types differ


def test_sensitive_values_are_masked_in_messages() -> None:
    with pytest.raises(AssertionError) as raised:
        assert_subset({"password": "actual-pw"}, {"password": "expected-pw"})

    assert "actual-pw" not in str(raised.value) and "expected-pw" not in str(raised.value)


def test_long_values_are_truncated() -> None:
    with pytest.raises(AssertionError) as raised:
        assert_subset({"text": "x" * 500}, {"text": "y"})

    assert "..." in str(raised.value) and len(str(raised.value)) < 300
