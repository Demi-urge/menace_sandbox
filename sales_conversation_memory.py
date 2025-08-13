import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple


class SalesConversationMemory:
    """Store recent sales conversation messages and CTA chain state.

    Only the message ``role`` and ``text`` are kept; timestamps are used
    internally for expiry and are not returned.
    """

    def __init__(self, max_messages: int = 6, ttl: float = 60.0) -> None:
        self.ttl = ttl
        # Store entries as (timestamp, {"role": role, "text": text})
        self._messages: Deque[Tuple[float, Dict[str, str]]] = deque(
            maxlen=max_messages
        )
        # Track a chain of call-to-action (CTA) events.
        self.cta_stack: List[Dict[str, Any]] = []

    def _purge(self, current_time: float | None = None) -> None:
        if current_time is None:
            current_time = time.time()
        while self._messages and current_time - self._messages[0][0] > self.ttl:
            self._messages.popleft()

    def add_message(
        self, text: str, role: str, timestamp: float | None = None
    ) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self._purge(ts)
        self._messages.append((ts, {"text": text, "role": role}))

    def push_cta(self, step: Dict[str, Any]) -> None:
        """Push a CTA step onto the stack."""
        self.cta_stack.append(step)

    def pop_cta(self) -> Dict[str, Any] | None:
        """Pop the most recent CTA step off the stack."""
        if self.cta_stack:
            return self.cta_stack.pop()
        return None

    def clear_cta_stack(self) -> None:
        """Clear the CTA stack, resolving or expiring the current chain."""
        self.cta_stack.clear()

    def get_recent(self) -> List[Dict[str, str]]:
        self._purge()
        return [msg for _, msg in self._messages]


__all__ = ["SalesConversationMemory"]
