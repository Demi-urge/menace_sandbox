from __future__ import annotations

"""Data store helpers for the self-improvement engine."""

from db_router import GLOBAL_ROUTER, init_db_router
from menace_sandbox.workflow_stability_db import WorkflowStabilityDB

router = GLOBAL_ROUTER or init_db_router("self_improvement")
STABLE_WORKFLOWS = WorkflowStabilityDB()

__all__ = ["router", "STABLE_WORKFLOWS"]
