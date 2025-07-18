from __future__ import annotations

"""Service running self tests on a schedule."""

import logging
import os
import shlex
import subprocess
import re
from threading import Event

from .cross_model_scheduler import _SimpleScheduler, BackgroundScheduler
from .error_bot import ErrorDB
from .error_logger import ErrorLogger


class SelfTestService:
    """Periodically execute the test suite to validate core bots."""

    def __init__(self, db: ErrorDB | None = None, *, pytest_args: str | None = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_logger = ErrorLogger(db)
        self.scheduler: object | None = None
        env_args = os.getenv("SELF_TEST_ARGS") if pytest_args is None else pytest_args
        self.pytest_args = shlex.split(env_args) if env_args else []

    # ------------------------------------------------------------------
    def _run_once(self) -> None:
        cmd = ["pytest", "-q"] + self.pytest_args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = (proc.stdout or "") + (proc.stderr or "")
        passed = 0
        failed = 0
        m = re.search(r"(\d+)\s+passed", output)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", output)
        if m:
            failed = int(m.group(1))
        try:
            self.error_logger.db.add_test_result(passed, failed)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("failed to store test results")
        try:
            proc.check_returncode()
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.error("self tests failed: %s", exc)
            try:
                self.error_logger.log(exc, "self_tests", "sandbox")
            except Exception:
                self.logger.exception("error logging failed")

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
