from __future__ import annotations

"""Watchdog to ensure ServiceSupervisor stays running."""

import logging
import os
import subprocess
import time
from typing import Iterable

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore


class SupervisorWatchdog:
    """Monitor the supervisor PID and restart when needed."""

    def __init__(
        self,
        pid_file: str = "/var/run/menace_supervisor.pid",
        start_cmd: Iterable[str] | None = None,
        check_interval: float = 10.0,
        restart_log: str = "supervisor_watchdog.log",
    ) -> None:
        self.pid_file = pid_file
        if start_cmd is None:
            start_cmd = ["python", "-m", "menace.service_supervisor"]
        self.start_cmd = list(start_cmd)
        self.check_interval = check_interval
        self.restart_log = restart_log
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _pid_alive(self, pid: int) -> bool:
        try:
            if psutil is not None:
                return psutil.pid_exists(pid)
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _restart(self) -> None:
        self.logger.warning("restarting ServiceSupervisor")
        try:
            subprocess.Popen(self.start_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(self.restart_log, "a", encoding="utf-8") as f:
                f.write(f"{time.time()}: restarted supervisor\n")
        except Exception as exc:  # pragma: no cover - subprocess may fail
            self.logger.error("failed to start supervisor: %s", exc)

    # ------------------------------------------------------------------
    def check(self) -> None:
        if not os.path.exists(self.pid_file):
            self._restart()
            return
        try:
            with open(self.pid_file, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
        except Exception:
            self.logger.error("invalid pid file")
            self._restart()
            return
        if not self._pid_alive(pid):
            self._restart()

    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            self.check()
            time.sleep(self.check_interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    pid_file = os.getenv("SUPERVISOR_PID_FILE", "/var/run/menace_supervisor.pid")
    interval = float(os.getenv("WATCHDOG_INTERVAL", "10"))
    cmd = os.getenv("SUPERVISOR_CMD", "python -m menace.service_supervisor").split()
    watch = SupervisorWatchdog(pid_file=pid_file, start_cmd=cmd, check_interval=interval)
    watch.run()


__all__ = ["SupervisorWatchdog", "main"]
