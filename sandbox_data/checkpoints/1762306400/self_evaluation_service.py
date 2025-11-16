from __future__ import annotations

"""Combine microtrend detection with workflow cloning."""

import logging
from threading import Event
from typing import Optional

from .microtrend_service import MicrotrendService
from .workflow_cloner import WorkflowCloner


class SelfEvaluationService:
    """Continuously assess trends and replicate successful workflows."""

    def __init__(
        self,
        microtrend: Optional[MicrotrendService] = None,
        cloner: Optional[WorkflowCloner] = None,
    ) -> None:
        self.microtrend = microtrend or MicrotrendService()
        self.cloner = cloner or WorkflowCloner()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _on_trend(self, items) -> None:
        try:
            limit = max(1, len(items)) if hasattr(items, "__len__") else 3
            self.cloner.clone_top_workflows(limit=limit)
        except Exception:
            self.logger.exception("clone failed")

    def run_continuous(self, interval: float = 3600.0, *, stop_event: Optional[Event] = None) -> None:
        self.microtrend.planner = self._on_trend
        self.cloner.start()
        self.microtrend.run_continuous(interval=interval, stop_event=stop_event)


__all__ = ["SelfEvaluationService"]
