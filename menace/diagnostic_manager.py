"""Compatibility re-exports for legacy ``menace.diagnostic_manager`` imports."""

from __future__ import annotations

from importlib import import_module

try:  # pragma: no cover - support package + flat layouts
    from .vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback for root imports
    from vector_service.context_builder import ContextBuilder  # type: ignore

__all__ = ["ResolutionRecord", "ResolutionDB", "DiagnosticManager", "ContextBuilder"]


def __getattr__(name: str):
    if name == "ContextBuilder":
        return ContextBuilder
    if name in {"ResolutionRecord", "ResolutionDB", "DiagnosticManager"}:
        module = import_module("diagnostic_manager")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
