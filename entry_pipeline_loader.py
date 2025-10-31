"""Runtime helpers for loading the automation pipeline implementation.

This module provides a single function that performs the heavyweight import
for :class:`menace_sandbox.shared.pipeline_base.ModelAutomationPipeline`.  It is
kept separate from :mod:`menace_sandbox.shared.model_pipeline_core` so that
imports within the ``shared`` package do not re-enter :mod:`pipeline_base`
while it is still initialising, preventing circular import failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from .shared.pipeline_base import ModelAutomationPipeline as _ModelAutomationPipeline
else:  # pragma: no cover - runtime fallback avoids circular import
    _ModelAutomationPipeline = Any  # type: ignore[misc, assignment]

__all__ = ["load_pipeline_class"]


def load_pipeline_class() -> "type[_ModelAutomationPipeline]":
    """Return the concrete :class:`ModelAutomationPipeline` implementation."""

    from .shared.pipeline_base import ModelAutomationPipeline as _Pipeline

    return _Pipeline
