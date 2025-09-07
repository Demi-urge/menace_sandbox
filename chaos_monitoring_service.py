from __future__ import annotations

"""Run chaos experiments and automatically rollback failing bots.

The service relies on :class:`vector_service.ContextBuilder` for contextual
analysis.  A builder must be provided by the caller when the default
scheduler is used.
"""

import logging
import threading
from threading import Event

from .chaos_scheduler import ChaosScheduler
from .watchdog import Watchdog
from .error_bot import ErrorDB
from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .advanced_error_management import AutomatedRollbackManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - optional dependency for type hints
    from vector_service import ContextBuilder


class ChaosMonitoringService:
    """Inject faults and rollback bots that fail to recover."""

    def __init__(
        self,
        scheduler: ChaosScheduler | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        if scheduler is None:
            if context_builder is None:
                raise ValueError("context_builder required when scheduler not provided")
            watch = Watchdog(ErrorDB(), ROIDB(), MetricsDB(), context_builder=context_builder)
            scheduler = ChaosScheduler(watchdog=watch)
        self.scheduler = scheduler
        self.rollback_mgr = rollback_mgr or AutomatedRollbackManager()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _monitor(self, stop: Event) -> None:
        while not stop.wait(self.scheduler.interval):
            for fault in list(self.scheduler.watchdog.synthetic_faults):
                if not fault.get("recovered"):
                    bot = fault.get("bot")
                    if bot:
                        try:
                            self.rollback_mgr.auto_rollback("latest", [bot])
                        except Exception:
                            self.logger.exception("auto rollback failed")

    def run_continuous(self, interval: float = 60.0, *, stop_event: Event | None = None) -> None:
        self.scheduler.interval = interval
        self.scheduler.start()
        stop = stop_event or Event()
        threading.Thread(target=self._monitor, args=(stop,), daemon=True).start()


__all__ = ["ChaosMonitoringService"]
