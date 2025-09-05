from __future__ import annotations

"""Watchdog to ensure ``menace_visual_agent_2.py`` stays running."""

import logging
import os
import time

from visual_agent_manager import VisualAgentManager
from dynamic_path_router import resolve_path

try:  # optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore


def _pid_alive(pid: int) -> bool:
    try:
        if psutil is not None:
            return psutil.pid_exists(pid)
        os.kill(pid, 0)
        return True
    except Exception:
        return False


class VisualAgentWatchdog:
    """Monitor the visual agent PID and restart when needed."""

    def __init__(
        self,
        manager: VisualAgentManager | None = None,
        *,
        check_interval: float = 10.0,
        restart_log: str = "visual_agent_watchdog.log",
    ) -> None:
        self.manager = manager or VisualAgentManager()
        self.check_interval = float(check_interval)
        try:
            self.restart_log = str(resolve_path(restart_log))
        except FileNotFoundError:
            self.restart_log = str(resolve_path(".") / restart_log)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _restart(self) -> None:
        self.logger.warning("restarting menace_visual_agent_2")
        try:
            token = os.getenv("VISUAL_AGENT_TOKEN", "")
            self.manager.restart_with_token(token)
            with open(self.restart_log, "a", encoding="utf-8") as f:
                f.write(f"{time.time()}: restarted visual agent\n")
        except Exception as exc:  # pragma: no cover - subprocess may fail
            self.logger.error("failed to start visual agent: %s", exc)

    # ------------------------------------------------------------------
    def check(self) -> None:
        pid_path = self.manager.pid_file
        if not pid_path.exists():
            self._restart()
            return
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            self.logger.error("invalid pid file")
            self._restart()
            return
        if not _pid_alive(pid):
            self._restart()

    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            self.check()
            time.sleep(self.check_interval)


def main() -> None:  # pragma: no cover - entrypoint
    logging.basicConfig(level=logging.INFO)
    interval = float(os.getenv("WATCHDOG_INTERVAL", "10"))
    wd = VisualAgentWatchdog(check_interval=interval)
    wd.run()


__all__ = ["VisualAgentWatchdog", "main"]
