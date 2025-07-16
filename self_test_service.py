from __future__ import annotations

"""Service running self tests on a schedule."""

import logging
import subprocess
from threading import Event

from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler


class SelfTestService:
    """Periodically execute the test suite to validate core bots."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scheduler: object | None = None

    # ------------------------------------------------------------------
    def _run_once(self) -> None:
        try:
            subprocess.run(["pytest", "-q"], check=True)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.error("self tests failed: %s", exc)

    # ------------------------------------------------------------------
    def run_continuous(self, interval: float = 86400.0, *, stop_event: Event | None = None) -> None:
        if self.scheduler:
            return
        if BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(self._run_once, "interval", seconds=interval, id="self_test")
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(self._run_once, interval, "self_test")
            self.scheduler = sched
        self._stop = stop_event or Event()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self.scheduler:
            return
        if hasattr(self, "_stop") and self._stop:
            self._stop.set()
        if BackgroundScheduler and isinstance(self.scheduler, BackgroundScheduler):
            self.scheduler.shutdown(wait=False)
        else:
            self.scheduler.shutdown()
        self.scheduler = None


__all__ = ["SelfTestService"]
