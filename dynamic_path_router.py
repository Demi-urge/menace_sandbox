from __future__ import annotations

"""Utilities for dynamically resolving paths within the repository.

The :func:`resolve_path` helper attempts to locate a file or directory by
name.  The repository root is determined via a series of fallbacks and the
filesystem may be crawled to locate the target.  Results from expensive
operations are cached to keep lookups fast on subsequent calls.
"""

from functools import lru_cache
from pathlib import Path
import os
import subprocess
from typing import Dict

# Cache of discovered paths.  Keys are normalized relative paths using POSIX
# separators.  Values are absolute :class:`Path` objects.
_INDEX_CACHE: Dict[str, Path] = {}
_INDEX_BUILT = False


def _normalize(name: str | Path) -> str:
    """Normalize *name* for cross-platform comparisons.

    Paths are converted to strings with POSIX separators so lookups are
    consistent across operating systems.  The returned value never contains
    a leading ``./``.
    """

    path = Path(str(name).replace("\\", "/"))
    return path.as_posix().lstrip("./")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root directory.

    Determination order:

    1. ``SANDBOX_REPO_PATH`` environment variable.
    2. ``git rev-parse --show-toplevel``.
    3. Upward search for a ``.git`` folder.
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

    current = Path.cwd().resolve()
    for folder in [current, *current.parents]:
        if (folder / ".git").exists():
            return folder

    raise FileNotFoundError("Repository root could not be determined")


def _build_index() -> None:
    """Populate the path index using ``os.walk``.

    This is executed at most once per process.  All discovered files and
    directories are stored in :data:`_INDEX_CACHE` under both their relative
    paths and their final component names.  The relative paths use POSIX
    separators to ensure portability.
    """

    global _INDEX_BUILT
    if _INDEX_BUILT:
        return

    root = repo_root()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            _INDEX_CACHE[rel] = path.resolve()
            _INDEX_CACHE.setdefault(name, path.resolve())

    _INDEX_BUILT = True


def resolve_path(name: str | Path) -> Path:
    """Resolve *name* to an absolute :class:`Path` within the repository.

    Parameters
    ----------
    name:
        File or directory to locate.  Can be a relative path, absolute path or
        just a base name.

    Returns
    -------
    pathlib.Path
        Normalised absolute path to the located file or directory.

    Raises
    ------
    FileNotFoundError
        If the target cannot be located.
    """

    key = _normalize(name)
    candidate = Path(key)

    if candidate.is_absolute():
        if candidate.exists():
            result = candidate.resolve()
            _INDEX_CACHE[key] = result
            return result
    else:
        direct = repo_root() / candidate
        if direct.exists():
            result = direct.resolve()
            _INDEX_CACHE[key] = result
            return result

    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    _build_index()

    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    for rel, path in _INDEX_CACHE.items():
        if rel.endswith(key):
            _INDEX_CACHE[key] = path
            return path

    raise FileNotFoundError(f"Unable to resolve path: {name}")


def resolve_dir(name: str | Path) -> Path:
    """Resolve *name* to a directory within the repository.

    Parameters
    ----------
    name:
        Directory to locate. Accepts the same forms as :func:`resolve_path`.

    Returns
    -------
    pathlib.Path
        Normalised absolute path to the located directory.

    Raises
    ------
    NotADirectoryError
        If the resolved path is not a directory.
    FileNotFoundError
        If the target cannot be located.
    """

    path = resolve_path(name)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory: {name}")
    return path


def clear_cache() -> None:
    """Clear internal caches used by this module."""

    _INDEX_CACHE.clear()
    global _INDEX_BUILT
    _INDEX_BUILT = False
    repo_root.cache_clear()
