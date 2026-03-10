from __future__ import annotations

"""Data store helpers for the self-improvement engine."""

from db_router import GLOBAL_ROUTER, init_db_router
from menace_sandbox.workflow_stability_db import WorkflowStabilityDB

router = GLOBAL_ROUTER or init_db_router("self_improvement")


class _WorkflowStabilityProxy:
    """Lazily instantiate ``WorkflowStabilityDB`` to avoid import-time I/O."""

    _db: WorkflowStabilityDB | None = None

    def _get(self) -> WorkflowStabilityDB:
        if self._db is None:
            self._db = WorkflowStabilityDB()
        return self._db

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


STABLE_WORKFLOWS = _WorkflowStabilityProxy()

__all__ = ["router", "STABLE_WORKFLOWS"]
