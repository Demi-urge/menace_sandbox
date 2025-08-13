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


class InteractionMemory:
    """Bounded queue storing recent interaction records.

    The underlying deque enforces a maximum length between 3 and 6 entries.
    """

    def __init__(self, maxlen: int = 6) -> None:
        if not 3 <= maxlen <= 6:
            raise ValueError("maxlen must be between 3 and 6")
        self._records: Deque[InteractionRecord] = deque(maxlen=maxlen)

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


__all__ = [
    "InteractionRecord",
    "InteractionMemory",
    "append_message",
    "recent_messages",
]
