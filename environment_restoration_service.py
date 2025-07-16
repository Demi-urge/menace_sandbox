from __future__ import annotations

"""Service to restore the runtime environment after crashes."""

import logging
from threading import Event
import subprocess

from .retry_utils import retry

from .environment_bootstrap import EnvironmentBootstrapper
from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler


class EnvironmentRestorationService:
    """Periodically re-run the bootstrapper to repair the environment."""

    def __init__(self, bootstrapper: EnvironmentBootstrapper | None = None) -> None:
        self.bootstrapper = bootstrapper or EnvironmentBootstrapper()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scheduler: object | None = None

    # ------------------------------------------------------------------
    @retry((RuntimeError, subprocess.CalledProcessError), attempts=3, delay=0.1)
    def _bootstrap(self) -> None:
        """Run the bootstrapper with retries."""
        self.bootstrapper.bootstrap()

    def _run_once(self) -> None:
        try:
            self._bootstrap()
        except (RuntimeError, subprocess.CalledProcessError) as exc:  # pragma: no cover - best effort
            self.logger.error("environment restoration failed: %s", exc)

    # ------------------------------------------------------------------
    def run_continuous(self, interval: float = 3600.0, *, stop_event: Event | None = None) -> None:
        if self.scheduler:
            return
        if BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(self._run_once, "interval", seconds=interval, id="env_restore")
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(self._run_once, interval, "env_restore")
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


__all__ = ["EnvironmentRestorationService"]


