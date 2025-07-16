"""Hierarchical escalation helpers with runbook tracking."""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass
class EscalationLevel:
    name: str
    notifier: object
    sla_minutes: int = 0
    response_minutes: int = 0


class EscalationProtocol:
    """Escalate alerts through multiple levels until acknowledged."""

    def __init__(self, levels: Iterable[EscalationLevel]):
        self.levels = list(levels)
        self.counter = itertools.count(1)

    def escalate(self, message: str, attachments: Iterable[str] | None = None) -> str:
        runbook_id = f"RB{next(self.counter):05d}"
        full = f"{runbook_id}: {message}"
        logger = logging.getLogger("EscalationProtocol")
        for level in self.levels:
            start = time.time()
            deadline = start + level.sla_minutes * 60 if level.sla_minutes else None
            logger.info("Escalating to %s at %s", level.name, datetime.utcnow().isoformat())
            acknowledged = False
            while True:
                try:
                    level.notifier.notify(full, attachments=attachments)
                except Exception:
                    logger.exception("Failed to notify %s", level.name)
                acknowledged = getattr(level.notifier, "acknowledged", True)
                if acknowledged:
                    break
                if deadline and time.time() >= deadline:
                    break
                interval = level.response_minutes * 60 if level.response_minutes else 60
                time.sleep(interval)
            if acknowledged:
                break
        return runbook_id


__all__ = ["EscalationProtocol", "EscalationLevel"]
