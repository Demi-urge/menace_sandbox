"""Utilities for resolving files within this repository.

The module exposes helpers for locating files, directories and modules across
one or more project roots.  Roots are determined by consulting optional
environment overrides (``MENACE_ROOT``/``SANDBOX_REPO_PATH`` or the multi-root
variants ``MENACE_ROOTS``/``SANDBOX_REPO_PATHS``), falling back to
``git rev-parse --show-toplevel`` and finally searching parent directories for
a ``.git`` directory.  When a direct lookup fails a full ``os.walk`` search is
used.  Successful lookups and discovered roots are cached behind a thread-safe
lock to avoid repeated filesystem walks.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import threading
from typing import Dict, List, Optional

# Cache of discovered paths keyed by "root:path" identifiers
_PATH_CACHE: Dict[str, Path] = {}
_PROJECT_ROOT: Optional[Path] = None
_PROJECT_ROOTS: Optional[List[Path]] = None
_CACHE_LOCK = threading.Lock()


def _normalize(name: str | Path) -> str:
    """Return *name* as a normalised POSIX style string."""

    return Path(str(name).replace("\\", "/")).as_posix().lstrip("./")


def _discover_roots(start: Optional[Path] = None) -> List[Path]:
    """Return a list of candidate repository roots."""

    for env_var in (
        "MENACE_ROOTS",
        "SANDBOX_REPO_PATHS",
        "MENACE_ROOT",
        "SANDBOX_REPO_PATH",
    ):
        env_path = os.environ.get(env_var)
        if env_path:
            roots: List[Path] = []
            for part in env_path.split(os.pathsep):
                if part:
                    path = Path(part).expanduser().resolve()
                    if path.exists():
                        roots.append(path)
            if roots:
                return roots

    root: Optional[Path] = None
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent

    try:
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(current),
        ).strip()
        if top_level:
            root = Path(top_level).resolve()
    except Exception:
        pass

    if root is None:
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists():
                root = parent
                break
        if root is None:
            root = current

    return [root]


def get_project_root(
    start: Optional[Path | str] = None, repo_hint: Optional[Path | str] = None
) -> Path:
    """Return a repository root directory.

    ``start`` specifies a starting directory for discovery. ``repo_hint`` may be
    a path or repository name when multiple roots are configured via the
    ``MENACE_ROOTS`` or ``SANDBOX_REPO_PATHS`` environment variables.
    Preference order for locating roots remains unchanged:

    1. ``MENACE_ROOTS`` or ``SANDBOX_REPO_PATHS`` environment variables.
    2. ``MENACE_ROOT`` or ``SANDBOX_REPO_PATH`` environment variables.
    3. ``git rev-parse --show-toplevel``.
    4. Upward search from ``start`` (or this file) for a ``.git`` directory.
    5. Directory containing this file.
    """

    global _PROJECT_ROOT, _PROJECT_ROOTS
    with _CACHE_LOCK:
        if _PROJECT_ROOTS is None:
            _PROJECT_ROOTS = _discover_roots(Path(start) if start else None)
            _PROJECT_ROOT = _PROJECT_ROOTS[0]

    roots = _PROJECT_ROOTS

    hint: Optional[Path] = None
    if repo_hint is not None:
        hint = Path(repo_hint).expanduser().resolve()
    elif start is not None:
        hint = Path(start).expanduser().resolve()

    if hint is not None:
        for r in roots:
            try:
                if hint == r or hint.is_relative_to(r):
                    return r
            except AttributeError:  # Python <3.9
                try:
                    hint.relative_to(r)
                    return r
                except Exception:
                    if r == hint:
                        return r
        for r in roots:
            if r.name == str(repo_hint):
                return r

    return roots[0]


# Backwards compatible aliases
project_root = get_project_root
repo_root = get_project_root


def _cache_key(root: Path, name: str) -> str:
    return f"{root.as_posix()}:{name}"


def get_project_roots() -> List[Path]:
    """Return the list of configured repository roots."""

    global _PROJECT_ROOT, _PROJECT_ROOTS
    with _CACHE_LOCK:
        if _PROJECT_ROOTS is None:
            _PROJECT_ROOTS = _discover_roots()
            _PROJECT_ROOT = _PROJECT_ROOTS[0]
        return list(_PROJECT_ROOTS)


def resolve_path(name: str | Path, root: Optional[Path | str] = None) -> Path:
    """Resolve *name* to an absolute :class:`Path` within configured roots."""

    name_str = str(name)

    # Automatically map Windows development paths to the Paperspace runtime layout.
    if "C:/cyberlolos_project" in name_str:
        name_str = name_str.replace("C:/cyberlolos_project", "/home/paperspace")
    if "C:\\cyberlolos_project" in name_str:
        name_str = name_str.replace("C:\\cyberlolos_project", "/home/paperspace")

    norm_name = _normalize(name_str)
    path = Path(name_str)
    if path.is_absolute():
        if path.exists():
            resolved = path.resolve()
            with _CACHE_LOCK:
                _PATH_CACHE[path.as_posix()] = resolved
            return resolved
        raise FileNotFoundError(f"{name!r} does not exist")

    roots = [get_project_root(repo_hint=root, start=root)] if root else get_project_roots()

    for base in roots:
        key = _cache_key(base, norm_name)
        with _CACHE_LOCK:
            cached = _PATH_CACHE.get(key)
        if cached is not None:
            return cached

        candidate = base / path
        if candidate.exists():
            resolved = candidate.resolve()
            with _CACHE_LOCK:
                _PATH_CACHE[key] = resolved
            return resolved

        # When the candidate's parent directory already exists we assume the
        # caller intends to create a new file at that location.  Returning the
        # prospective path avoids an expensive full repository walk looking for
        # a match that does not yet exist (which can stall startup when large
        # directory trees are scanned).
        if candidate.parent.exists():
            resolved = candidate.resolve()
            with _CACHE_LOCK:
                _PATH_CACHE[key] = resolved
            return resolved

    target = Path(norm_name)
    for base in roots:
        for dirpath, dirs, files in os.walk(base):
            if ".git" in dirs:
                dirs.remove(".git")
            if target.name in files or target.name in dirs:
                match = Path(dirpath) / target.name
                rel = match.relative_to(base).as_posix()
                if rel.endswith(norm_name):
                    resolved = match.resolve()
                    with _CACHE_LOCK:
                        _PATH_CACHE[_cache_key(base, norm_name)] = resolved
                    return resolved

    roots_str = ", ".join(r.as_posix() for r in roots)
    raise FileNotFoundError(f"{name!r} not found under {roots_str}")


def resolve_module_path(module_name: str) -> Path:
    """Resolve dotted ``module_name`` to a Python source file across all roots."""

    module_path = Path(*module_name.split("."))
    try:
        return resolve_path(module_path.with_suffix(".py").as_posix())
    except FileNotFoundError:
        return resolve_path(str(module_path / "__init__.py"))


def resolve_dir(dirname: str) -> Path:
    """Resolve *dirname* to a directory across the configured roots."""

    path = resolve_path(dirname)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory: {dirname}")
    return path


def path_for_prompt(name: str) -> str:
    """Return a normalised string path suitable for inclusion in prompts."""

    return resolve_path(name).as_posix()


def clear_cache() -> None:
    """Clear internal caches used by this module."""

    global _PROJECT_ROOT, _PROJECT_ROOTS
    with _CACHE_LOCK:
        _PATH_CACHE.clear()
        _PROJECT_ROOT = None
        _PROJECT_ROOTS = None


def list_files() -> Dict[str, Path]:
    """Return a copy of the internal cache mapping."""

    with _CACHE_LOCK:
        return dict(_PATH_CACHE)


__all__ = [
    "get_project_root",
    "get_project_roots",
    "resolve_path",
    "resolve_module_path",
    "resolve_dir",
    "path_for_prompt",
    "project_root",
    "repo_root",
    "clear_cache",
    "list_files",
]
