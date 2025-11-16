from __future__ import annotations

"""Service performing periodic secret rotation."""

import logging
import os
from threading import Event
from typing import Sequence

from .secrets_manager import SecretsManager
from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler


class SecretRotationService:
    """Periodically rotate configured secrets."""

    def __init__(
        self,
        manager: SecretsManager | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        self.manager = manager or SecretsManager()
        env_names = os.getenv("ROTATE_SECRET_NAMES", "")
        env_list = [n.strip() for n in env_names.split(",") if n.strip()]
        self.names = list(names or env_list)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scheduler: object | None = None

    # ------------------------------------------------------------------
    def run_once(self) -> None:
        for name in self.names:
            try:
                self.manager.get(name, rotate=True)
            except Exception:
                self.logger.exception("rotation failed for %s", name)

    # ------------------------------------------------------------------
    def run_continuous(
        self,
        interval: float = 86400.0,
        *,
        stop_event: Event | None = None,
    ) -> None:
        """Start the scheduler."""

        if self.scheduler:
            return
        if BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(self.run_once, "interval", seconds=interval, id="secret_rotation")
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(self.run_once, interval, "secret_rotation")
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


__all__ = ["SecretRotationService"]
