import time
from collections import deque
from typing import Deque, Dict, List


class SalesConversationMemory:
    """Store recent sales conversation messages with time-based expiration."""

    def __init__(self, max_messages: int = 6, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self.messages: Deque[Dict[str, float | str]] = deque(maxlen=max_messages)

    def _purge(self, current_time: float | None = None) -> None:
        if current_time is None:
            current_time = time.time()
        while self.messages and current_time - self.messages[0]["timestamp"] > self.ttl:
            self.messages.popleft()

    def add_message(self, text: str, role: str, timestamp: float | None = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self._purge(ts)
        self.messages.append({"text": text, "role": role, "timestamp": ts})

    def get_recent(self) -> List[Dict[str, float | str]]:
        self._purge()
        return list(self.messages)


__all__ = ["SalesConversationMemory"]
