"""Listener that keeps :mod:`evaluation_dashboard` lineage trees up to date."""

from __future__ import annotations

import logging
from typing import Any

from .unified_event_bus import UnifiedEventBus
from .mutation_logger import build_lineage
from .evaluation_dashboard import EvaluationDashboard

logger = logging.getLogger(__name__)


class LineageDashboardListener:
    """Subscribe to mutation events and update an :class:`EvaluationDashboard`."""

    def __init__(
        self,
        dashboard: EvaluationDashboard,
        event_bus: UnifiedEventBus | None = None,
    ) -> None:
        self.dashboard = dashboard
        self.event_bus = event_bus or UnifiedEventBus()
        self.event_bus.subscribe("mutation_recorded", self._handle)

    # ------------------------------------------------------------------
    def _handle(self, _topic: str, payload: Any) -> None:
        """Handle mutation events by rebuilding the affected lineage tree."""

        try:
            if isinstance(payload, dict):
                workflow_id = payload.get("workflow_id")
                if workflow_id is None:
                    return
                tree = build_lineage(int(workflow_id))
                self.dashboard.update_lineage_tree(int(workflow_id), tree)
        except Exception:
            logger.exception("failed processing mutation_recorded event")


__all__ = ["LineageDashboardListener"]
