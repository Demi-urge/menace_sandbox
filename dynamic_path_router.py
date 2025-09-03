"""Utilities for resolving files within this repository.

The module provides :func:`resolve_path` which locates files relative to the
project root.  The root is determined by consulting an optional environment
override (``MENACE_ROOT`` or the legacy ``SANDBOX_REPO_PATH``), falling back to
``git rev-parse --show-toplevel`` and finally searching parent directories for a
``.git`` directory.  When a direct lookup fails a full ``os.walk`` search is
used.  Successful lookups are cached to avoid repeated scans.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Dict, Optional

# Cache of discovered paths keyed by normalised POSIX style names
_PATH_CACHE: Dict[str, Path] = {}
_PROJECT_ROOT: Optional[Path] = None


def _normalize(name: str | Path) -> str:
    """Return *name* as a normalised POSIX style string."""

    return Path(str(name).replace("\\", "/")).as_posix().lstrip("./")


def get_project_root() -> Path:
    """Return the repository root directory.

    Preference order:

    1. ``MENACE_ROOT`` or legacy ``SANDBOX_REPO_PATH`` environment variable.
    2. ``git rev-parse --show-toplevel``.
    3. Upward search from this file for a ``.git`` directory.
    4. Directory containing this file.
    """

    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    env_path = os.environ.get("MENACE_ROOT") or os.environ.get(
        "SANDBOX_REPO_PATH"
    )
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            _PROJECT_ROOT = path
            return path

    try:
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if top_level:
            _PROJECT_ROOT = Path(top_level).resolve()
            return _PROJECT_ROOT
    except Exception:
        pass

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            _PROJECT_ROOT = parent
            return parent

    _PROJECT_ROOT = current
    return _PROJECT_ROOT


# Backwards compatible aliases
project_root = get_project_root
repo_root = get_project_root


def resolve_path(name: str) -> Path:
    """Resolve *name* to an absolute :class:`Path` within the repository."""

    key = _normalize(name)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]

    path = Path(name)
    if path.is_absolute():
        if path.exists():
            resolved = path.resolve()
            _PATH_CACHE[key] = resolved
            return resolved
        raise FileNotFoundError(f"{name!r} does not exist")

    root = get_project_root()
    candidate = root / path
    if candidate.exists():
        resolved = candidate.resolve()
        _PATH_CACHE[key] = resolved
        return resolved

    target = Path(key)
    for base, dirs, files in os.walk(root):
        if ".git" in dirs:
            dirs.remove(".git")
        if target.name in files or target.name in dirs:
            match = Path(base) / target.name
            rel = match.relative_to(root).as_posix()
            if rel.endswith(key):
                resolved = match.resolve()
                _PATH_CACHE[key] = resolved
                return resolved

    raise FileNotFoundError(f"{name!r} not found under {root}")


def resolve_module_path(module_name: str) -> Path:
    """Resolve dotted ``module_name`` to a Python source file."""

    module_path = Path(*module_name.split("."))
    try:
        return resolve_path(module_path.with_suffix(".py").as_posix())
    except FileNotFoundError:
        return resolve_path(str(module_path / "__init__.py"))


def resolve_dir(dirname: str) -> Path:
    """Resolve *dirname* to a directory within the repository."""

    path = resolve_path(dirname)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory: {dirname}")
    return path


def path_for_prompt(name: str) -> str:
    """Return a normalised string path suitable for inclusion in prompts."""

    return str(resolve_path(name))


def clear_cache() -> None:
    """Clear internal caches used by this module."""

    _PATH_CACHE.clear()
    global _PROJECT_ROOT
    _PROJECT_ROOT = None


def list_files() -> Dict[str, Path]:
    """Return a copy of the internal cache mapping."""

    return dict(_PATH_CACHE)


__all__ = [
    "get_project_root",
    "resolve_path",
    "resolve_module_path",
    "resolve_dir",
    "path_for_prompt",
    "project_root",
    "repo_root",
    "clear_cache",
    "list_files",
]
