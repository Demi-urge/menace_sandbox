"""Utilities for dynamically resolving paths within the repository.

This module exposes :func:`resolve_path` which attempts to locate files or
directories relative to the project root.  The root is determined by invoking
``git rev-parse --show-toplevel`` and falling back to walking the parents of
this file.  Results are cached to avoid repeated filesystem searches.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import subprocess
from typing import Dict

# Cache of previously discovered paths.  Keys use POSIX style separators to
# keep lookups platform independent.
_PATH_CACHE: Dict[str, Path] = {}


def _normalize(name: str | Path) -> str:
    """Return *name* as a normalised POSIX style string."""

    return Path(str(name).replace("\\", "/")).as_posix().lstrip("./")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root directory.

    Preference order:

    1. ``SANDBOX_REPO_PATH`` environment variable.
    2. ``git rev-parse --show-toplevel``.
    3. Upward search from this file for a ``.git`` directory.
    """

    env_path = os.environ.get("SANDBOX_REPO_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    try:
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if top_level:
            return Path(top_level).resolve()
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent

    raise FileNotFoundError("Repository root could not be determined")


def resolve_path(filename: str) -> Path:
    """Resolve *filename* to an absolute :class:`Path` within the repository.

    Searches the repository root for ``filename``.  Direct path joins are
    attempted first and a recursive glob search is used as a fallback.  Results
    are cached to avoid repeated filesystem walks.
    """

    key = _normalize(filename)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]

    path = Path(filename)
    if path.is_absolute():
        if path.exists():
            resolved = path.resolve()
            _PATH_CACHE[key] = resolved
            return resolved
        raise FileNotFoundError(f"'{filename}' does not exist")

    root = repo_root()
    candidate = root / path
    if candidate.exists():
        resolved = candidate.resolve()
        _PATH_CACHE[key] = resolved
        return resolved

    # Fallback to rglob search when direct lookup fails.
    for match in root.rglob(path.name):
        rel = match.relative_to(root).as_posix()
        if rel.endswith(key):
            resolved = match.resolve()
            _PATH_CACHE[key] = resolved
            return resolved

    raise FileNotFoundError(
        f"{filename!r} not found under repository root {repo_root()}"
    )


def resolve_dir(dirname: str) -> Path:
    """Resolve *dirname* to a directory within the repository."""

    path = resolve_path(dirname)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory: {dirname}")
    return path


def clear_cache() -> None:
    """Clear internal caches used by this module."""

    _PATH_CACHE.clear()
    repo_root.cache_clear()


__all__ = ["resolve_path", "resolve_dir", "repo_root", "clear_cache"]

