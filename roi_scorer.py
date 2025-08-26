from __future__ import annotations

"""Backward compatibility shim for workflow scoring utilities.

This module previously contained a heavy implementation of
:class:`CompositeWorkflowScorer`.  The implementation now lives in
``composite_workflow_scorer`` and is re-exported here to keep legacy import
paths functioning while avoiding the additional dependencies of the old
module.
"""

from .composite_workflow_scorer import (
    ROIScorer,
    CompositeWorkflowScorer,
)
from .workflow_scorer_core import (
    Scorecard,
    compute_workflow_synergy,
    compute_bottleneck_index,
    compute_patchability,
)

__all__ = [
    "ROIScorer",
    "CompositeWorkflowScorer",
    "Scorecard",
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
]
