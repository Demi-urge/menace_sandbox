from __future__ import annotations

"""Run chaos experiments and automatically rollback failing bots."""

import logging
import threading
from threading import Event

from .chaos_scheduler import ChaosScheduler
from .watchdog import Watchdog, get_default_context_builder
from .error_bot import ErrorDB
from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .advanced_error_management import AutomatedRollbackManager


class ChaosMonitoringService:
    """Inject faults and rollback bots that fail to recover."""

    def __init__(self,
                 scheduler: ChaosScheduler | None = None,
                 rollback_mgr: AutomatedRollbackManager | None = None) -> None:
        builder = get_default_context_builder()
        watch = Watchdog(ErrorDB(), ROIDB(), MetricsDB(), context_builder=builder)
        self.scheduler = scheduler or ChaosScheduler(watchdog=watch)
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
