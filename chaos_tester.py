"""Utilities for injecting failures to test resilience."""

from __future__ import annotations

import random
import subprocess
import threading
import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


class ChaosTester:
    """Simulate hardware failures and network disruptions."""

    def chaos_monkey(
        self,
        processes: Optional[Iterable[subprocess.Popen]] = None,
        threads: Optional[Iterable[threading.Thread]] = None,
    ) -> Optional[str]:
        """Randomly kill a subprocess or suspend a thread.

        Parameters
        ----------
        processes:
            Iterable of ``subprocess.Popen`` instances that may be killed.
        threads:
            Iterable of ``threading.Thread`` instances that may be suspended.

        Returns
        -------
        str | None
            Description of the action taken or ``None`` if nothing happened.
        """
        choices = []
        if processes:
            choices.append("proc")
        if threads:
            choices.append("thread")
        if not choices:
            return None
        action = random.choice(choices)
        if action == "proc":
            proc = random.choice(list(processes))
            try:
                proc.terminate()
            except Exception:
                logger.exception("failed to terminate process")
            return "killed_process"
        else:
            thread = random.choice(list(threads))
            evt = getattr(thread, "suspend_event", None)
            if isinstance(evt, threading.Event):
                evt.set()
            return "suspended_thread"

    @staticmethod
    def drop_db_connection(conn) -> None:
        """Drop a database connection."""
        try:
            conn.close()
        except Exception:
            logger.warning("failed to close database connection", exc_info=True)

    @staticmethod
    def corrupt_network_data(data: bytes) -> bytes:
        """Return corrupted bytes of the same length."""
        if not data:
            return data
        idx = random.randrange(len(data))
        corrupted = bytearray(data)
        corrupted[idx] = (corrupted[idx] + random.randint(1, 255)) % 256
        return bytes(corrupted)

    @staticmethod
    def corrupt_disk(path: str) -> None:
        """Simulate disk corruption by writing random bytes."""
        try:
            with open(path, "r+b") as f:
                f.seek(0)
                data = f.read()
                if data:
                    pos = random.randrange(len(data))
                    f.seek(pos)
                    f.write(bytes([random.randint(0, 255)]))
        except Exception as exc:
            logger.warning("failed to corrupt disk %s: %s", path, exc, exc_info=True)

    @staticmethod
    def partition_network(hosts: Iterable[str]) -> list[str]:
        """Return list of hosts to block communication with."""
        blocked = []
        for h in hosts:
            if random.random() < 0.5:
                blocked.append(h)
        return blocked

    @staticmethod
    def validate_recovery(bot, action) -> bool:
        """Run *action* and check if the bot triggered rollback or recovery."""
        try:
            action()
        except Exception:
            logger.exception("chaos action failed")
        return any(
            getattr(bot, attr, False) for attr in ("rolled_back", "recovered")
        )

