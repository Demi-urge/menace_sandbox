"""Compatibility wrapper for the ModelAutomationPipeline implementation.

This module exists so that call sites can import
``menace_sandbox.shared.model_pipeline_core`` without triggering the heavy
initialisation performed by :mod:`menace_sandbox.shared.execution_core` during
module import.  The direct import previously performed here caused a circular
import when modules such as :mod:`capital_management_bot` lazily imported the
automation pipeline.  By deferring the import until attribute access we avoid
initialising :mod:`execution_core` while the importing module is still being
constructed, breaking the cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from .execution_core import ModelAutomationPipeline as _ModelAutomationPipeline
else:  # pragma: no cover - runtime fallback avoids circular import
    _ModelAutomationPipeline = Any  # type: ignore[misc, assignment]

_PIPELINE_CLS: type[_ModelAutomationPipeline] | None = None

__all__ = ["ModelAutomationPipeline"]


def _load_pipeline_cls() -> type[_ModelAutomationPipeline]:
    """Import and cache the concrete pipeline implementation."""

    global _PIPELINE_CLS
    if _PIPELINE_CLS is None:
        from .execution_core import ModelAutomationPipeline as _Pipeline

        _PIPELINE_CLS = _Pipeline
        globals()["ModelAutomationPipeline"] = _PIPELINE_CLS
    return _PIPELINE_CLS


def __getattr__(name: str) -> Any:
    """Expose :class:`ModelAutomationPipeline` lazily to avoid circular imports."""

    if name == "ModelAutomationPipeline":
        return _load_pipeline_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Ensure ``dir()`` includes lazily provided attributes."""

    return sorted(list(globals().keys()) + ["ModelAutomationPipeline"])
