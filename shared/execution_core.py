"""Compatibility wrapper for :mod:`menace_sandbox.shared.pipeline_base`.

Historically the :class:`ModelAutomationPipeline` implementation lived in this
module.  It now resides in :mod:`menace_sandbox.shared.pipeline_base` so that
other parts of the system can import a neutral definition without triggering
the heavier dependencies required by the execution core.  This module re-exports
the class to preserve backwards compatibility with existing import sites.
"""

from __future__ import annotations

from .pipeline_base import ModelAutomationPipeline

__all__ = ["ModelAutomationPipeline"]

