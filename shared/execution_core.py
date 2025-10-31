"""Compatibility wrapper for :mod:`menace_sandbox.shared.pipeline_base`.

Historically the :class:`ModelAutomationPipeline` implementation lived in this
module.  It now resides in :mod:`menace_sandbox.shared.pipeline_base` so that
other parts of the system can import a neutral definition without triggering
the heavier dependencies required by the execution core.  This module re-exports
the class to preserve backwards compatibility with existing import sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only import avoids circular dependency
    from .pipeline_base import ModelAutomationPipeline  # noqa: F401

__all__: Final = ["ModelAutomationPipeline"]


def __getattr__(name: str) -> Any:
    """Dynamically import :class:`ModelAutomationPipeline` on first access."""

    if name != "ModelAutomationPipeline":
        raise AttributeError(name)

    from .pipeline_base import ModelAutomationPipeline as _Pipeline

    globals()[name] = _Pipeline
    return _Pipeline

