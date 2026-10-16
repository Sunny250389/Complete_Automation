"""Artifact storage interface and a local file-system implementation.

Artifacts are written under a root directory (normally ``<artifacts.root>/<run_id>``).
Names are sanitised, paths cannot leave the root, files are created exclusively
(never overwritten), and every write returns an :class:`ArtifactRef` that
reports can cite. Contents are stored as given; a module that writes sensitive
diagnostics must redact them first.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from complete_automation.core.context import utc_now
from complete_automation.core.exceptions import ArtifactError
from complete_automation.core.paths import UnsafePathError, resolve_within
from complete_automation.core.serialization import dumps_deterministic

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME_ATTEMPTS = 1000


class ArtifactRef(BaseModel):
    """Where an artifact was stored, and enough metadata to verify it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    path: str = Field(description="POSIX path relative to the store root.")
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


@runtime_checkable
class ArtifactStore(Protocol):
    """What every artifact store provides. Future adapters (object storage) implement this."""

    def write_bytes(
        self,
        name: str,
        data: bytes,
        *,
        subdir: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef: ...

    def write_text(
        self,
        name: str,
        text: str,
        *,
        subdir: str | None = None,
        media_type: str = "text/plain",
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef: ...

    def write_json(
        self,
        name: str,
        data: Any,
        *,
        subdir: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef: ...


class LocalArtifactStore:
    """Stores artifacts as files under ``root``. The directory is created on first write."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    @classmethod
    def for_run(cls, artifacts_root: Path, run_id: str) -> LocalArtifactStore:
        return cls(Path(artifacts_root) / sanitize_name(run_id))

    def write_bytes(
        self,
        name: str,
        data: bytes,
        *,
        subdir: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef:
        if not isinstance(data, (bytes, bytearray)):
            raise ArtifactError(f"write_bytes expects bytes, got {type(data).__name__}")
        directory = self._directory(subdir)
        target = self._create_exclusive(directory, sanitize_name(name), bytes(data))
        return ArtifactRef(
            name=name,
            path=target.relative_to(self.root).as_posix(),
            media_type=media_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            created_at=utc_now(),
            metadata=dict(metadata or {}),
        )

    def write_text(
        self,
        name: str,
        text: str,
        *,
        subdir: str | None = None,
        media_type: str = "text/plain",
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef:
        return self.write_bytes(
            name, text.encode("utf-8"), subdir=subdir, media_type=media_type, metadata=metadata
        )

    def write_json(
        self,
        name: str,
        data: Any,
        *,
        subdir: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> ArtifactRef:
        try:
            text = dumps_deterministic(data)
        except (TypeError, ValueError) as error:
            raise ArtifactError(f"artifact {name!r} is not JSON-serializable: {error}") from error
        return self.write_text(
            name, text, subdir=subdir, media_type="application/json", metadata=metadata
        )

    def resolve(self, ref: ArtifactRef) -> Path:
        """Absolute path of a stored artifact; refuses references outside this store."""
        try:
            return resolve_within(self.root, ref.path)
        except UnsafePathError as error:
            raise ArtifactError(str(error)) from error

    def _directory(self, subdir: str | None) -> Path:
        if subdir is None or subdir in ("", "."):
            return self.root
        try:
            return resolve_within(self.root, subdir)
        except UnsafePathError as error:
            raise ArtifactError(f"unsafe artifact subdirectory: {error}") from error

    def _create_exclusive(self, directory: Path, name: str, data: bytes) -> Path:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArtifactError(f"cannot create artifact directory {directory}") from error
        stem, dot, suffix = name.partition(".")
        for attempt in range(_MAX_NAME_ATTEMPTS):
            candidate = directory / (name if attempt == 0 else f"{stem}-{attempt}{dot}{suffix}")
            try:
                with candidate.open("xb") as handle:
                    handle.write(data)
                return candidate
            except FileExistsError:
                continue
            except OSError as error:
                raise ArtifactError(f"cannot write artifact {candidate}") from error
        raise ArtifactError(f"no free artifact name for {name!r} in {directory}")


def sanitize_name(name: str) -> str:
    """Make ``name`` a safe single path segment: ``[A-Za-z0-9._-]`` only, no leading dots."""
    cleaned = _UNSAFE_CHARS.sub("_", name.strip()).lstrip(".")
    if not cleaned or set(cleaned) <= {"_"}:
        raise ArtifactError(f"artifact name has no usable characters: {name!r}")
    return cleaned[:200]
