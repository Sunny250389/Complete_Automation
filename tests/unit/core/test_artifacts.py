import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from complete_automation.core.artifacts import ArtifactRef, ArtifactStore, LocalArtifactStore, sanitize_name
from complete_automation.core.exceptions import ArtifactError
from complete_automation.core.results import Status, TestResult


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts" / "run-1")


def test_write_text_returns_a_verifiable_reference(store: LocalArtifactStore) -> None:
    ref = store.write_text("notes.txt", "héllo", metadata={"step": "login"})

    path = store.resolve(ref)
    assert path.read_text(encoding="utf-8") == "héllo"
    assert ref.path == "notes.txt"
    assert ref.size_bytes == len("héllo".encode())
    assert ref.sha256 == hashlib.sha256("héllo".encode()).hexdigest()
    assert ref.media_type == "text/plain"
    assert ref.metadata == {"step": "login"}


def test_nested_directories_are_created(store: LocalArtifactStore) -> None:
    ref = store.write_bytes("shot.png", b"\x89PNG", subdir="ui/test_login", media_type="image/png")

    assert ref.path == "ui/test_login/shot.png"
    assert store.resolve(ref).read_bytes() == b"\x89PNG"


def test_existing_artifacts_are_never_overwritten(store: LocalArtifactStore) -> None:
    first = store.write_text("report.json.txt", "first")
    second = store.write_text("report.json.txt", "second")

    assert first.path == "report.json.txt"
    assert second.path == "report-1.json.txt"
    assert store.resolve(first).read_text(encoding="utf-8") == "first"


def test_json_artifacts_are_deterministic(store: LocalArtifactStore) -> None:
    ref = store.write_json("eval.json", {"b": 1, "a": [datetime(2026, 1, 1, tzinfo=UTC)]})

    text = store.resolve(ref).read_text(encoding="utf-8")
    assert ref.media_type == "application/json"
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text)["a"] == ["2026-01-01T00:00:00+00:00"]


def test_non_serializable_json_is_an_error(store: LocalArtifactStore) -> None:
    with pytest.raises(ArtifactError, match="not JSON-serializable"):
        store.write_json("bad.json", {"handle": object()})


@pytest.mark.parametrize("subdir", ["../escape", "a/../../escape", "/etc", "C:/Windows"])
def test_subdirectories_cannot_escape_the_root(store: LocalArtifactStore, subdir: str) -> None:
    with pytest.raises(ArtifactError):
        store.write_text("x.txt", "x", subdir=subdir)


def test_names_are_sanitised_into_the_root(store: LocalArtifactStore) -> None:
    ref = store.write_text("../../evil name?.txt", "x")

    assert "/" not in ref.path and not ref.path.startswith(".")
    assert store.resolve(ref).parent == store.root


def test_forged_references_cannot_escape(store: LocalArtifactStore) -> None:
    forged = ArtifactRef(
        name="x", path="../../outside.txt", media_type="text/plain", size_bytes=0,
        sha256="0" * 64, created_at=datetime.now(UTC),
    )

    with pytest.raises(ArtifactError):
        store.resolve(forged)


@pytest.mark.parametrize("name", ["", "...", "///", "  "])
def test_unusable_names_are_rejected(name: str) -> None:
    with pytest.raises(ArtifactError):
        sanitize_name(name)


def test_write_failures_raise_artifact_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not directory", encoding="utf-8")

    with pytest.raises(ArtifactError, match="cannot create artifact directory") as raised:
        LocalArtifactStore(blocker).write_text("x.txt", "x")
    assert isinstance(raised.value.__cause__, OSError)


def test_bytes_are_required_for_binary_writes(store: LocalArtifactStore) -> None:
    with pytest.raises(ArtifactError):
        store.write_bytes("x.bin", "text")  # type: ignore[arg-type]


def test_run_store_is_created_lazily(tmp_path: Path) -> None:
    store = LocalArtifactStore.for_run(tmp_path, "20261016T000000Z-abcd1234")

    assert store.root == (tmp_path / "20261016T000000Z-abcd1234").resolve()
    assert not store.root.exists()
    store.write_text("a.txt", "a")
    assert store.root.is_dir()


def test_references_attach_to_results(store: LocalArtifactStore) -> None:
    ref = store.write_text("trace.txt", "trace")
    now = datetime.now(UTC)

    result = TestResult(
        run_id="run-1", test_id="t", name="t", module="ui", status=Status.PASSED,
        started_at=now, finished_at=now, artifacts=(ref,),
    )

    assert result.to_dict()["artifacts"][0]["path"] == "trace.txt"


def test_local_store_satisfies_the_protocol(store: LocalArtifactStore) -> None:
    assert isinstance(store, ArtifactStore)
