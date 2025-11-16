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
        # Store queue entries as ``(timestamp, {"role": role, "text": text})``
        self._messages: Deque[Tuple[float, Dict[str, str]]] = deque(
            maxlen=max_messages
        )
        # Track a chain of call-to-action (CTA) events with timestamps.
        self._cta_stack: List[Tuple[float, Dict[str, Any]]] = []

    def prune(
        self, decay_seconds: float | None = None, current_time: float | None = None
    ) -> None:
        """Remove queue and stack entries older than ``decay_seconds``."""

        if current_time is None:
            current_time = time.time()
        window = decay_seconds if decay_seconds is not None else self.ttl
        cutoff = current_time - window
        while self._messages and self._messages[0][0] < cutoff:
            self._messages.popleft()
        self._cta_stack = [s for s in self._cta_stack if s[0] >= cutoff]

    def add_message(
        self, text: str, role: str, timestamp: float | None = None
    ) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self.prune(current_time=ts)
        self._messages.append((ts, {"text": text, "role": role}))

    def push_cta(self, step: Dict[str, Any], timestamp: float | None = None) -> None:
        """Push a CTA step onto the stack."""

        ts = timestamp if timestamp is not None else time.time()
        self.prune(current_time=ts)
        self._cta_stack.append((ts, step))

    def pop_cta(self) -> Dict[str, Any] | None:
        """Pop the most recent CTA step off the stack."""

        self.prune()
        if self._cta_stack:
            return self._cta_stack.pop()[1]
        return None

    def clear_cta_stack(self) -> None:
        """Clear the CTA stack, resolving or expiring the current chain."""

        self._cta_stack.clear()

    @property
    def cta_stack(self) -> List[Dict[str, Any]]:
        """Return CTA stack entries without timestamps."""

        self.prune()
        return [step for _, step in self._cta_stack]

    def get_recent(self) -> List[Dict[str, str]]:
        self.prune()
        return [msg for _, msg in self._messages]


__all__ = ["SalesConversationMemory"]
