"""Thread-local guard tracking active self-coding module imports.

This utility prevents recursive bot health checks from re-importing modules
while they are still being defined. Both :mod:`coding_bot_interface` and
:mod:`bot_registry` use the helpers to coordinate initial registration during
bootstrap without deadlocking on import-level side effects.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "self_coding_import_guard",
    "is_self_coding_import_active",
    "self_coding_import_depth",
]


_IMPORT_STATE = threading.local()


def _normalise(value: os.PathLike[str] | str | None) -> str | None:
    """Return a stable string representation for *value*."""

    if value is None:
        return None
    try:
        # ``Path`` handles both ``str`` and ``os.PathLike`` inputs. Normalising
        # to ``resolve()`` keeps comparisons stable across relative inputs while
        # tolerating missing files by falling back to ``absolute()`` semantics.
        return str(Path(value).resolve())
    except Exception:
        # ``module_path`` may point to logical module names which are not valid
        # filesystem paths. Fall back to ``str`` so we can still track the value
        # in the guard stack.
        return str(value)


@contextmanager
def self_coding_import_guard(
    module_path: os.PathLike[str] | str | None,
) -> Iterator[None]:
    """Record that *module_path* is being imported by a coding bot."""

    normalised = _normalise(module_path)
    stack: list[str | None] = list(getattr(_IMPORT_STATE, "stack", []))
    stack.append(normalised)
    _IMPORT_STATE.stack = stack
    try:
        yield
    finally:
        try:
            stack.pop()
        finally:
            if stack:
                _IMPORT_STATE.stack = stack
            elif hasattr(_IMPORT_STATE, "stack"):
                delattr(_IMPORT_STATE, "stack")


def is_self_coding_import_active(
    module_path: os.PathLike[str] | str | None = None,
) -> bool:
    """Return ``True`` when a self-coding import is currently active."""

    stack: Iterable[str | None] | None = getattr(_IMPORT_STATE, "stack", None)
    if not stack:
        return False
    if module_path is None:
        return True
    target = _normalise(module_path)
    return any(entry == target for entry in stack if entry is not None)


def self_coding_import_depth(
    module_path: os.PathLike[str] | str | None = None,
) -> int:
    """Return the nesting depth for the active self-coding import."""

    stack: Iterable[str | None] | None = getattr(_IMPORT_STATE, "stack", None)
    if not stack:
        return 0
    if module_path is None:
        return len(list(stack))
    target = _normalise(module_path)
    return sum(1 for entry in stack if entry == target)
