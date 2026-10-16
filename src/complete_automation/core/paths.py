"""Path helpers shared by configuration, artifacts and test data."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

PROJECT_MARKER = "pyproject.toml"


class UnsafePathError(ValueError):
    """A path would escape the directory it must stay inside."""


def find_project_root(start: Path | None = None) -> Path | None:
    """Return the nearest directory at or above ``start`` holding ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_MARKER).is_file():
            return candidate
    return None


def resolve_within(root: Path, relative: str | Path) -> Path:
    """Join ``relative`` onto ``root`` and refuse any result outside ``root``.

    Absolute paths, drive-qualified paths and ``..`` segments that climb out of
    ``root`` raise :class:`UnsafePathError`. The returned path is absolute.
    """
    text = str(relative)
    if not text.strip():
        raise UnsafePathError("path must not be empty")
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise UnsafePathError(f"path must be relative: {text!r}")
    if PureWindowsPath(text).drive:
        raise UnsafePathError(f"path must not name a drive: {text!r}")

    base = root.resolve()
    candidate = (base / text).resolve()
    if not candidate.is_relative_to(base):
        raise UnsafePathError(f"path escapes {base}: {text!r}")
    return candidate
