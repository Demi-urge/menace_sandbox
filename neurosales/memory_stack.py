from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence


@dataclass
class CTAChain:
    """Represents a single CTA exchange in the sales conversation."""

    message_ts: datetime
    reply_ts: datetime
    escalation_ts: datetime
    created_at: datetime


class MemoryStack:
    """Maintain a stack of recent CTA chains with optional expiry."""

    def __init__(
        self,
        max_size: int = 10,
        decay_seconds: Optional[int] = 3600,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self.decay_seconds = decay_seconds
        self._stack: List[CTAChain] = []

    # ------------------------------------------------------------------
    def _expire_old_records(self, now: Optional[datetime] = None) -> None:
        if self.decay_seconds is None:
            return
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.decay_seconds)
        self._stack = [chain for chain in self._stack if chain.created_at >= cutoff]

    # ------------------------------------------------------------------
    def push_chain(
        self,
        message_ts: datetime,
        reply_ts: datetime,
        escalation_ts: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
    ) -> CTAChain:
        created = created_at or datetime.now(timezone.utc)
        self._expire_old_records(created)
        chain = CTAChain(
            message_ts=message_ts,
            reply_ts=reply_ts,
            escalation_ts=escalation_ts or created,
            created_at=created,
        )
        self._stack.append(chain)
        if len(self._stack) > self.max_size:
            self._stack = self._stack[-self.max_size :]
        return chain

    # ------------------------------------------------------------------
    def peek_chain(self) -> Optional[CTAChain]:
        self._expire_old_records()
        if not self._stack:
            return None
        return self._stack[-1]

    # ------------------------------------------------------------------
    def pop_chain(self) -> Optional[CTAChain]:
        self._expire_old_records()
        if not self._stack:
            return None
        return self._stack.pop()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._stack.clear()

    # ------------------------------------------------------------------
    def current_chain(self) -> Sequence[CTAChain]:
        self._expire_old_records()
        return list(self._stack)


_default_stack = MemoryStack()


def _coerce_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    ts = getattr(value, "timestamp", None)
    if isinstance(ts, datetime):
        return ts
    raise TypeError("value must be a datetime or expose a 'timestamp' attribute")


def push_chain(
    message: object,
    reply: object,
    created_at: Optional[datetime] = None,
) -> CTAChain:
    """Push a CTA interaction onto the default stack."""

    message_ts = _coerce_timestamp(message)
    reply_ts = _coerce_timestamp(reply)
    created = created_at or datetime.now(timezone.utc)
    return _default_stack.push_chain(message_ts, reply_ts, escalation_ts=created, created_at=created)


def peek_chain() -> Optional[CTAChain]:
    """Return the most recent CTA chain without removing it."""

    return _default_stack.peek_chain()


def pop_chain() -> Optional[CTAChain]:
    """Remove and return the most recent CTA chain."""

    return _default_stack.pop_chain()


def clear_stack() -> None:
    """Clear the global CTA chain stack."""

    _default_stack.clear()


__all__ = [
    "CTAChain",
    "MemoryStack",
    "push_chain",
    "peek_chain",
    "pop_chain",
    "clear_stack",
]
