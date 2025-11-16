from __future__ import annotations

"""Periodic dependency update service."""

import logging
import subprocess
import os
from datetime import datetime
from threading import Event
from pathlib import Path

from .dependency_update_bot import DependencyUpdater
from typing import Optional
from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler, _AsyncScheduler


class DependencyUpdateService:
    """Run :class:`DependencyUpdater` on a schedule."""

    def __init__(self, updater: DependencyUpdater | None = None, db: Optional[object] = None) -> None:
        self.updater = updater or DependencyUpdater()
        if db is None:
            from .deployment_bot import DeploymentDB  # lazy import
            db = DeploymentDB()
        self.db = db
        self.logger = logging.getLogger("DependencyUpdateService")
        self.scheduler: object | None = None

    # ------------------------------------------------------------------
    def _run_once(self, *, verify_host: str | None = None) -> None:
        packages: list[str] = []
        try:
            packages = self.updater.run_cycle(update_os=True)
            if not packages:
                return
            subprocess.run(["uv", "pip", "compile", "pyproject.toml"], check=True)
            subprocess.run(["pytest", "-q"], check=True)
            subprocess.run(["git", "pull", "origin", "main"], check=True)
            subprocess.run(["git", "add", "uv.lock"], check=True)
            subprocess.run(["git", "commit", "-m", "Auto update"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            if Path("Dockerfile").exists():
                try:
                    subprocess.run([
                        "docker",
                        "build",
                        "-t",
                        "menace:latest",
                        ".",
                    ], check=True)
                    subprocess.run([
                        "docker",
                        "compose",
                        "up",
                        "-d",
                        "--quiet-pull",
                    ], check=False)
                    host = verify_host or os.getenv("DEP_VERIFY_HOST")
                    if host:
                        subprocess.run([
                            "ssh",
                            host,
                            "docker",
                            "run",
                            "--rm",
                            "menace:latest",
                            "pytest",
                            "-q",
                        ], check=True)
                except Exception as exc:
                    self.logger.error("container update failed: %s", exc)
            self.db.add_update(packages, "success")
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.error("update failed: %s", exc)
            self.db.add_update(packages, "failed")

    # ------------------------------------------------------------------
    def run_continuous(self, interval: float = 86400.0, *, stop_event: Event | None = None) -> None:
        """Start the scheduler."""

        if self.scheduler:
            return
        use_async = os.getenv("USE_ASYNC_SCHEDULER")
        if use_async:
            sched = _AsyncScheduler()
            sched.add_job(self._run_once, interval, "dep_update")
            self.scheduler = sched
        elif BackgroundScheduler:
            sched = BackgroundScheduler()
            sched.add_job(self._run_once, "interval", seconds=interval, id="dep_update")
            sched.start()
            self.scheduler = sched
        else:
            sched = _SimpleScheduler()
            sched.add_job(self._run_once, interval, "dep_update")
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


__all__ = ["DependencyUpdateService"]
