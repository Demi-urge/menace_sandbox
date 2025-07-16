"""Run ChaosTester periodically to inject failures."""

from __future__ import annotations

import threading
import time
from typing import Iterable, Optional

from .chaos_tester import ChaosTester
from .watchdog import Watchdog


class ChaosScheduler:
    """Periodically execute ``ChaosTester.chaos_monkey`` on targets."""

    def __init__(
        self,
        *,
        processes: Optional[Iterable[threading.Thread]] = None,
        threads: Optional[Iterable[threading.Thread]] = None,
        bots: Optional[Iterable[object]] = None,
        disk_paths: Optional[Iterable[str]] = None,
        hosts: Optional[Iterable[str]] = None,
        interval: int = 60,
        tester: Optional[ChaosTester] = None,
        watchdog: Optional[Watchdog] = None,
    ) -> None:
        self.processes = list(processes or [])
        self.threads = list(threads or [])
        self.bots = list(bots or [])
        self.disk_paths = list(disk_paths or [])
        self.hosts = list(hosts or [])
        self.interval = interval
        self.tester = tester or ChaosTester()
        self.watchdog = watchdog
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def _record(self, action: str) -> None:
        """Helper to log *action* via the watchdog."""
        if not self.watchdog:
            return
        for bot in self.bots:
            check = getattr(bot, "check", None)
            if callable(check):
                self.watchdog.record_fault(action, bot, check)
            else:
                self.watchdog.record_fault(action)
        if not self.bots:
            self.watchdog.record_fault(action)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            action = self.tester.chaos_monkey(
                processes=self.processes, threads=self.threads
            )
            if action:
                self._record(action)

            for path in self.disk_paths:
                self.tester.corrupt_disk(path)
                self._record("corrupt_disk")

            if self.hosts:
                blocked = self.tester.partition_network(self.hosts)
                if blocked:
                    self._record("partition_network")
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None


__all__ = ["ChaosScheduler"]
