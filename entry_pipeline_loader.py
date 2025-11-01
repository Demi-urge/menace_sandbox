"""Runtime helpers for loading the automation pipeline implementation.

This module provides a single function that performs the heavyweight import
for :class:`menace_sandbox.shared.pipeline_base.ModelAutomationPipeline`.  It is
kept separate from :mod:`menace_sandbox.shared.model_pipeline_core` so that
imports within the ``shared`` package do not re-enter :mod:`pipeline_base`
while it is still initialising, preventing circular import failures.
"""

from __future__ import annotations

import importlib
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from .shared.pipeline_base import ModelAutomationPipeline as _ModelAutomationPipeline
else:  # pragma: no cover - runtime fallback avoids circular import
    _ModelAutomationPipeline = Any  # type: ignore[misc, assignment]

__all__ = ["load_pipeline_class"]


def _resolve_pipeline_module() -> Any:
    """Import ``pipeline_base`` handling partially initialised modules."""

    module_name = "menace_sandbox.shared.pipeline_base"
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "ModelAutomationPipeline", None) is not None:
        return module

    module = importlib.import_module(module_name)
    pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
    if pipeline_cls is not None:
        return module

    # The module is present but still initialising.  Wait briefly for the
    # attribute to appear before attempting a clean re-import.  This situation
    # occurs when ``pipeline_base`` is imported indirectly while one of its
    # dependencies is still importing ``capital_management_bot``.
    for _ in range(10):
        time.sleep(0.05)
        pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
        if pipeline_cls is not None:
            return module

    # Last resort: remove the partially initialised module and import again.
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return module


def load_pipeline_class() -> "type[_ModelAutomationPipeline]":
    """Return the concrete :class:`ModelAutomationPipeline` implementation."""

    module = _resolve_pipeline_module()
    pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
    if pipeline_cls is None:
        raise ImportError("ModelAutomationPipeline is unavailable")
    return pipeline_cls
