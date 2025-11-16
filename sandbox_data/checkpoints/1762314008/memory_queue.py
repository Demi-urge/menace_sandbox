from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, List, Optional


@dataclass
class MessageEntry:
    """Representation of a single conversation message."""

    text: str
    role: str = "assistant"
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class MemoryQueue:
    """Bounded queue that stores recent conversation messages."""

    def __init__(
        self,
        max_size: int = 20,
        decay_seconds: Optional[int] = 3600,
    ) -> None:
        if max_size < 3:
            raise ValueError("max_size must be at least 3 to retain context")
        self.max_size = max_size
        self.decay_seconds = decay_seconds
        self._queue: Deque[MessageEntry] = deque(maxlen=max_size)

    # ------------------------------------------------------------------
    def _expire_old_records(self, now: Optional[datetime] = None) -> None:
        if self.decay_seconds is None:
            return
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.decay_seconds)
        while self._queue and (self._queue[0].timestamp or now) < cutoff:
            self._queue.popleft()

    # ------------------------------------------------------------------
    def add_message(
        self,
        text: str,
        role: str = "assistant",
        timestamp: Optional[datetime] = None,
    ) -> MessageEntry:
        self._expire_old_records(timestamp)
        entry = MessageEntry(text=text, role=role, timestamp=timestamp)
        self._queue.append(entry)
        return entry

    # ------------------------------------------------------------------
    def get_recent_messages(self, limit: Optional[int] = None) -> List[MessageEntry]:
        self._expire_old_records()
        messages = list(self._queue)
        if limit is None:
            return messages
        if limit <= 0:
            return []
        return messages[-limit:]

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._queue.clear()


_default_queue = MemoryQueue()


def add_message(
    text: str,
    role: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> MessageEntry:
    """Add a message to the default queue used by the conversation manager."""

    return _default_queue.add_message(text=text, role=role or "assistant", timestamp=timestamp)


def get_recent_messages(limit: int = 5) -> List[MessageEntry]:
    """Return the most recent ``limit`` messages from the default queue."""

    return _default_queue.get_recent_messages(limit=limit)


__all__ = ["MessageEntry", "MemoryQueue", "add_message", "get_recent_messages"]
