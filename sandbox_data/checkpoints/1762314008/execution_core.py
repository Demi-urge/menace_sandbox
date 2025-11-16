"""Compatibility wrapper for :mod:`menace_sandbox.shared.pipeline_base`.

Historically the :class:`ModelAutomationPipeline` implementation lived in this
module.  It now resides in :mod:`menace_sandbox.shared.pipeline_base` so that
other parts of the system can import a neutral definition without triggering
the heavier dependencies required by the execution core.  This module re-exports
the class to preserve backwards compatibility with existing import sites.
"""

from __future__ import annotations

print(">>> [trace] Entered execution_core.py")
print(">>> [trace] Successfully imported annotations from __future__")

print(">>> [trace] Importing TYPE_CHECKING, Any, Final from typing...")
from typing import TYPE_CHECKING, Any, Final
print(">>> [trace] Successfully imported TYPE_CHECKING, Any, Final from typing")

if TYPE_CHECKING:  # pragma: no cover - typing only import avoids circular dependency
    print(">>> [trace] Importing ModelAutomationPipeline for type checking from menace_sandbox.shared.pipeline_base...")
    from .pipeline_base import ModelAutomationPipeline  # noqa: F401
    print(">>> [trace] Successfully imported ModelAutomationPipeline for type checking from menace_sandbox.shared.pipeline_base")

__all__: Final = ["ModelAutomationPipeline"]


def __getattr__(name: str) -> Any:
    """Dynamically import :class:`ModelAutomationPipeline` on first access."""

    if name != "ModelAutomationPipeline":
        raise AttributeError(name)

    print(">>> [trace] Lazily importing ModelAutomationPipeline from menace_sandbox.shared.pipeline_base...")
    from .pipeline_base import ModelAutomationPipeline as _Pipeline
    print(">>> [trace] Successfully imported ModelAutomationPipeline from menace_sandbox.shared.pipeline_base")

    globals()[name] = _Pipeline
    return _Pipeline

