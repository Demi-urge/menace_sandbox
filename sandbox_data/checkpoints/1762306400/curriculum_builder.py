from __future__ import annotations

"""Generate training curriculum from frequent telemetry errors."""

from typing import List, Dict

import logging

from .error_bot import ErrorBot
from .unified_event_bus import EventBus
from .error_flags import RAISE_ERRORS

logger = logging.getLogger(__name__)


class CurriculumBuilder:
    """Build datasets of common failure types for incremental training."""

    def __init__(self, error_bot: ErrorBot, event_bus: EventBus, *, threshold: int = 3) -> None:
        self.error_bot = error_bot
        self.event_bus = event_bus
        self.threshold = threshold

    def build(self) -> List[Dict[str, str]]:
        """Return curriculum entries for frequent errors."""
        summary = self.error_bot.summarize_telemetry()
        curriculum: List[Dict[str, str]] = []
        for item in summary:
            if item.get("count", 0.0) >= self.threshold:
                curriculum.append({"error_type": str(item.get("error_type", ""))})
        return curriculum

    def publish(self) -> List[Dict[str, str]]:
        """Publish curriculum entries as events."""
        items = self.build()
        for entry in items:
            try:
                self.event_bus.publish("curriculum:new", entry)
            except Exception as exc:
                logger.exception("publish failed: %s", exc)
                if RAISE_ERRORS:
                    raise
        return items


__all__ = ["CurriculumBuilder"]
