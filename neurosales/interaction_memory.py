from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Sequence


@dataclass
class InteractionRecord:
    """Record of a single conversational exchange."""

    message: str
    role: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)


@dataclass
class CTAEvent:
    """Metadata for a single Call-To-Action outcome."""

    message: str
    escalation_level: int = 0
    success: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionMemory:
    """Bounded queue storing recent interaction records.

    The underlying deque enforces a maximum length between 3 and 6 entries.
    """

    def __init__(self, maxlen: int = 6) -> None:
        if not 3 <= maxlen <= 6:
            raise ValueError("maxlen must be between 3 and 6")
        self._records: Deque[InteractionRecord] = deque(maxlen=maxlen)
        self._cta_stack: List[CTAEvent] = []

    def append_message(
        self,
        message: str,
        role: str,
        timestamp: datetime | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        """Append a message to the memory queue."""

        record = InteractionRecord(
            message=message,
            role=role,
            timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
            tags=list(tags) if tags is not None else [],
        )
        self._records.append(record)

    def push_event(
        self,
        message: str,
        escalation_level: int = 0,
        success: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Push a CTA event onto the internal stack."""

        event = CTAEvent(
            message=message,
            escalation_level=escalation_level,
            success=success,
            timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
        )
        self._cta_stack.append(event)

    def pop_chain(self) -> List[CTAEvent]:
        """Pop and return the current CTA chain."""

        chain = self._cta_stack
        self._cta_stack = []
        return chain

    def current_chain(self) -> List[CTAEvent]:
        """Return a copy of the active CTA chain."""

        return list(self._cta_stack)

    def recent_messages(
        self,
        include_triggers: bool = True,
        include_objections: bool = True,
    ) -> List[InteractionRecord]:
        """Return recent messages with optional filtering.

        ``include_triggers`` and ``include_objections`` control whether records
        tagged with ``"trigger"`` or ``"objection"`` are returned.
        """

        records = list(self._records)
        if include_triggers and include_objections:
            return records

        filtered: List[InteractionRecord] = []
        for rec in records:
            if not include_triggers and "trigger" in rec.tags:
                continue
            if not include_objections and "objection" in rec.tags:
                continue
            filtered.append(rec)
        return filtered


_default_memory = InteractionMemory()


def append_message(
    message: str,
    role: str,
    timestamp: datetime | None = None,
    tags: Sequence[str] | None = None,
) -> None:
    """Append a message to the default interaction memory."""

    _default_memory.append_message(
        message=message, role=role, timestamp=timestamp, tags=tags
    )


def recent_messages(
    include_triggers: bool = True, include_objections: bool = True
) -> List[InteractionRecord]:
    """Retrieve recent messages from the default interaction memory."""

    return _default_memory.recent_messages(
        include_triggers=include_triggers, include_objections=include_objections
    )


def push_event(
    message: str,
    escalation_level: int = 0,
    success: bool = False,
    timestamp: datetime | None = None,
) -> None:
    """Push a CTA event onto the default interaction memory stack."""

    _default_memory.push_event(
        message=message,
        escalation_level=escalation_level,
        success=success,
        timestamp=timestamp,
    )


def pop_chain() -> List[CTAEvent]:
    """Pop the current CTA chain from the default interaction memory."""

    return _default_memory.pop_chain()


def current_chain() -> List[CTAEvent]:
    """Return the active CTA chain from the default interaction memory."""

    return _default_memory.current_chain()


__all__ = [
    "InteractionRecord",
    "InteractionMemory",
    "append_message",
    "recent_messages",
    "CTAEvent",
    "push_event",
    "pop_chain",
    "current_chain",
]
