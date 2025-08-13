from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    The queue keeps at most ``max_size`` entries (between 3 and 6).
    Older entries are automatically discarded when the limit is reached.
    """

    def __init__(self, max_size: int = 6) -> None:
        if not 3 <= max_size <= 6:
            raise ValueError("max_size must be between 3 and 6")
        self._queue: Deque[MessageEntry] = deque(maxlen=max_size)

    def add_message(
        self, text: str, trigger: str | None = None, objection: str | None = None
    ) -> None:
        """Add a message to the queue with optional metadata."""

        self._queue.append(
            MessageEntry(text=text, trigger=trigger, objection=objection)
        )

    def get_recent_messages(self, count: int | None = None) -> List[MessageEntry]:
        """Return up to ``count`` of the most recent messages.

        Messages are ordered from oldest to newest.
        """

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
