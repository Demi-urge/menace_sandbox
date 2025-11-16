"""Utilities for generating defensive representations of objects.

These helpers avoid triggering potentially expensive or recursive
``__repr__`` implementations during diagnostic logging.  They prefer
simple type-oriented summaries which remain stable even when the object
graph contains cycles.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = [
    "safe_repr",
    "summarise_value",
    "basic_repr",
]


def _is_simple_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def _truncate(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarise_value(value: Any) -> str:
    """Return a short description of ``value`` without calling ``repr``.

    The summary favours type information over content to ensure that
    recursive structures cannot trigger unbounded recursion.
    """

    if value is None:
        return "None"
    if isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return f"{value.__class__.__name__}({_truncate(value)!r})"
    if isinstance(value, bytes):
        return f"bytes(len={len(value)})"
    if isinstance(value, dict):
        return f"{value.__class__.__name__}(len={len(value)})"
    if _is_simple_sequence(value):
        return f"{value.__class__.__name__}(len={len(value)})"
    if isinstance(value, Iterable):
        return f"{value.__class__.__name__}(iterable)"
    return f"{value.__class__.__name__}(id=0x{id(value):x})"


def basic_repr(obj: Any, *, attrs: dict[str, Any] | None = None) -> str:
    """Construct a defensive representation for ``obj``.

    ``attrs`` is an optional mapping of attribute names to values.  Each
    value is summarised via :func:`summarise_value` to avoid cascading
    ``repr`` calls on nested objects.
    """

    parts: list[str] = []
    if attrs:
        for key, value in attrs.items():
            parts.append(f"{key}={summarise_value(value)}")
    suffix = f" {' '.join(parts)}" if parts else ""
    return f"<{obj.__class__.__name__}{suffix} at 0x{id(obj):x}>"


def safe_repr(obj: Any) -> str:
    """Return ``repr(obj)`` but guard against failures.

    When ``repr`` raises an exception the fallback mirrors the input
    object's type and identifier to keep logging functional without
    introducing further errors.
    """

    try:
        return repr(obj)
    except Exception:  # pragma: no cover - defensive fallback
        return f"<{obj.__class__.__name__} id=0x{id(obj):x}>"

