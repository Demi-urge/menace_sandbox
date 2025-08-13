from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List

from .memory_queue import MessageEntry


@dataclass
class CTAChain:
    """Record linking a message, reply and escalation.

    Each field stores the timestamp of the corresponding entry in the
    :class:`~neurosales.memory_queue.MemoryQueue`.
    """

    message_ts: datetime
    reply_ts: datetime
    escalation_ts: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryStack:
    """Simple LIFO stack tracking CTA chains with expiry."""

    def __init__(self, decay_seconds: float = 300) -> None:
        self.decay = timedelta(seconds=decay_seconds)
        self._stack: List[CTAChain] = []

    def _expire_old_chains(self) -> None:
        """Drop chains older than the decay window."""

        cutoff = datetime.now(timezone.utc) - self.decay
        self._stack = [c for c in self._stack if c.created_at >= cutoff]

    @staticmethod
    def _get_ts(entry: MessageEntry | datetime) -> datetime:
        """Return the timestamp for ``entry``.

        ``entry`` may be a :class:`MessageEntry` or a ``datetime`` directly.
        """

        return entry.timestamp if isinstance(entry, MessageEntry) else entry

    def push_chain(
        self,
        message: MessageEntry | datetime,
        reply: MessageEntry | datetime,
        escalation: MessageEntry | datetime,
    ) -> None:
        """Push a ``message → reply → escalation`` chain onto the stack.

        The chain links to entries in :mod:`neurosales.memory_queue` via the
        timestamps of the individual messages.
        """

        self._expire_old_chains()
        self._stack.append(
            CTAChain(
                message_ts=self._get_ts(message),
                reply_ts=self._get_ts(reply),
                escalation_ts=self._get_ts(escalation),
            )
        )

    def peek_chain(self) -> CTAChain | None:
        """Return the most recently added chain without removing it."""
        self._expire_old_chains()
        return self._stack[-1] if self._stack else None

    def pop_chain(self) -> CTAChain | None:
        """Remove and return the most recent chain."""
        self._expire_old_chains()
        return self._stack.pop() if self._stack else None


_default_stack = MemoryStack()


def push_chain(
    message: MessageEntry | datetime,
    reply: MessageEntry | datetime,
    escalation: MessageEntry | datetime,
) -> None:
    """Push a chain onto the default stack."""

    _default_stack.push_chain(message=message, reply=reply, escalation=escalation)


def peek_chain() -> CTAChain | None:
    """Peek at the most recent chain on the default stack."""

    return _default_stack.peek_chain()


def pop_chain() -> CTAChain | None:
    """Pop the most recent chain from the default stack."""

    return _default_stack.pop_chain()


__all__ = [
    "CTAChain",
    "MemoryStack",
    "push_chain",
    "peek_chain",
    "pop_chain",
]
