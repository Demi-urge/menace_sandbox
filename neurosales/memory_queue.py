from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, List


@dataclass
class MessageEntry:
    """Record of a single message interaction."""

    text: str
    trigger: str | None = None
    objection: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryQueue:
    """Simple bounded FIFO queue for storing recent messages.

    The queue keeps at most ``max_size`` entries (between 3 and 6).  Entries
    older than ``decay_seconds`` are pruned on access.
    """

    def __init__(self, max_size: int = 6, decay_seconds: float = 300) -> None:
        if not 3 <= max_size <= 6:
            raise ValueError("max_size must be between 3 and 6")
        self.decay = timedelta(seconds=decay_seconds)
        self._queue: Deque[MessageEntry] = deque(maxlen=max_size)

    def _expire_old_entries(self) -> None:
        """Remove messages older than the decay window."""

        cutoff = datetime.now(timezone.utc) - self.decay
        while self._queue and self._queue[0].timestamp < cutoff:
            self._queue.popleft()

    def add_message(
        self, text: str, trigger: str | None = None, objection: str | None = None
    ) -> None:
        """Add a message to the queue with optional metadata."""
        self._expire_old_entries()
        self._queue.append(
            MessageEntry(text=text, trigger=trigger, objection=objection)
        )

    def get_recent_messages(self, count: int | None = None) -> List[MessageEntry]:
        """Return up to ``count`` of the most recent messages.

        Messages are ordered from oldest to newest.
        """
        self._expire_old_entries()
        if count is None or count >= len(self._queue):
            return list(self._queue)
        return list(self._queue)[-count:]


_default_queue = MemoryQueue()


def add_message(text: str, trigger: str | None = None, objection: str | None = None) -> None:
    """Add a message to the default memory queue."""

    _default_queue.add_message(text=text, trigger=trigger, objection=objection)


def get_recent_messages(count: int | None = None) -> List[MessageEntry]:
    """Retrieve messages from the default memory queue."""

    return _default_queue.get_recent_messages(count)


__all__ = [
    "MessageEntry",
    "MemoryQueue",
    "add_message",
    "get_recent_messages",
]
