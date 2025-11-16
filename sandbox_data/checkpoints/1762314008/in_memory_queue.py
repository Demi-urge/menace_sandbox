from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from queue import Queue
from threading import Thread


@dataclass
class QueuedTask:
    """Container for a queued task and its creation time."""

    name: str
    kwargs: dict | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryQueue:
    """Very small asynchronous queue for local task dispatch with expiry."""

    def __init__(self, decay_seconds: float = 300) -> None:
        self.decay = timedelta(seconds=decay_seconds)
        self.queue: Queue[QueuedTask] = Queue()
        self.sent: list[QueuedTask] = []
        self.executed: list[QueuedTask] = []
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def _expire_old_records(self) -> None:
        """Drop entries older than the decay window from bookkeeping lists."""

        cutoff = datetime.now(timezone.utc) - self.decay
        self.sent = [t for t in self.sent if t.created_at >= cutoff]
        self.executed = [t for t in self.executed if t.created_at >= cutoff]

    def send_task(self, name: str, kwargs: dict | None = None) -> None:
        self._expire_old_records()
        task = QueuedTask(name=name, kwargs=kwargs)
        self.sent.append(task)
        self.queue.put(task)

    def _run(self) -> None:
        while True:
            task = self.queue.get()
            cutoff = datetime.now(timezone.utc) - self.decay
            if task.created_at >= cutoff:
                self.executed.append(task)
            self.queue.task_done()
            self._expire_old_records()


__all__ = ["InMemoryQueue", "QueuedTask"]
