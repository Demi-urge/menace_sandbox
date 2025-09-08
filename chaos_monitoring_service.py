from __future__ import annotations

"""Run chaos experiments and automatically rollback failing bots.

The service relies on :class:`vector_service.ContextBuilder` for contextual
analysis.  Callers must supply a builder instance and database weights are
refreshed at startup to ensure ranking reflects the latest metrics.
"""

import logging
import threading
from threading import Event
from typing import TYPE_CHECKING

from .chaos_scheduler import ChaosScheduler
from .watchdog import Watchdog
from .error_bot import ErrorDB
from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .advanced_error_management import AutomatedRollbackManager

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - surface early
    ContextBuilder = None  # type: ignore


class ChaosMonitoringService:
    """Inject faults and rollback bots that fail to recover."""

    def __init__(
        self,
        scheduler: ChaosScheduler | None = None,
        rollback_mgr: AutomatedRollbackManager | None = None,
        *,
        context_builder: ContextBuilder,
    ) -> None:
        """Create service.

        Parameters
        ----------
        scheduler:
            Optional pre-configured scheduler. When omitted a default scheduler
            using ``context_builder`` is created.
        rollback_mgr:
            Automated rollback manager.
        context_builder:
            Builder used for contextual analysis.
        """
        if ContextBuilder is None or not isinstance(context_builder, ContextBuilder):
            raise TypeError("context_builder must be a ContextBuilder instance")
        try:
            context_builder.refresh_db_weights()
        except Exception as exc:  # pragma: no cover - surface configuration issues
            raise RuntimeError("failed to refresh database weights") from exc
        if scheduler is None:
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
