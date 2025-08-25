from __future__ import annotations

try:
    from .databases import MenaceDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore

__all__ = ['MenaceDB']

