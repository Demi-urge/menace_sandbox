from __future__ import annotations

"""Run chaos experiments and automatically rollback failing bots.

The service relies on :class:`vector_service.ContextBuilder` for contextual
analysis.  Callers must supply a builder instance and database weights are
refreshed at startup to ensure ranking reflects the latest metrics.
"""

import logging
import threading
from importlib import import_module
from threading import Event
from typing import Any, TYPE_CHECKING

from .chaos_scheduler import ChaosScheduler
from .watchdog import Watchdog
from .error_bot import ErrorDB
from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .advanced_error_management import AutomatedRollbackManager

def _resolve_context_builder_class() -> type[Any] | None:
    """Return the canonical :class:`ContextBuilder` class if available."""

    for module_path in (
        "vector_service.context_builder",
        "menace_sandbox.vector_service.context_builder",
    ):
        try:
            module = import_module(module_path)
        except Exception:
            continue
        builder_cls = getattr(module, "ContextBuilder", None)
        if isinstance(builder_cls, type):
            return builder_cls
    return None


ContextBuilder = _resolve_context_builder_class()


def resolve_context_builder(value: Any) -> Any:
    """Normalise a worker-provided builder or builder factory input."""

    candidate = value() if callable(value) else value
    if ContextBuilder is None:
        return candidate
    if isinstance(candidate, ContextBuilder):
        return candidate
    if (
        candidate is not None
        and candidate.__class__.__name__ == "ContextBuilder"
        and candidate.__class__.__module__.endswith("vector_service.context_builder")
    ):
        kwargs = {
            attr: getattr(candidate, attr)
            for attr in ("bots_db", "code_db", "errors_db", "workflows_db")
            if hasattr(candidate, attr)
        }
        if kwargs:
            return ContextBuilder(**kwargs)
    return candidate


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
            received_cls = context_builder.__class__
            raise TypeError(
                "context_builder must be a ContextBuilder instance; "
                f"received {received_cls.__module__}.{received_cls.__name__}"
            )
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
