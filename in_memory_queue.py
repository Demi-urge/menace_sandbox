from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Any, Tuple


class InMemoryQueue:
    """Very small asynchronous queue for local task dispatch."""

    def __init__(self) -> None:
        self.queue: Queue[Tuple[str, dict | None]] = Queue()
        self.sent: list[Tuple[str, dict | None]] = []
        self.executed: list[Tuple[str, dict | None]] = []
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def send_task(self, name: str, kwargs: dict | None = None) -> None:
        self.sent.append((name, kwargs))
        self.queue.put((name, kwargs))

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            self.executed.append(item)
            self.queue.task_done()


__all__ = ["InMemoryQueue"]
