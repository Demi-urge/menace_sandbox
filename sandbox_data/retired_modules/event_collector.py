from __future__ import annotations

"""Unified event collector for tracking bot activity.

``EventCollector`` subscribes to one or more topics on a
:class:`~unified_event_bus.UnifiedEventBus` and records relationships between
bots and workflows.  By default it listens for ``bot:call`` and
``workflows:new`` style events, updating the :class:`~bot_registry.BotRegistry`
accordingly.  When supplied with a :class:`~neuroplasticity.PathwayDB` each
event is also logged as a :class:`~neuroplasticity.PathwayRecord` for later
analysis.

Typical usage::

    bus = UnifiedEventBus()
    collector = EventCollector(bus, topics=["bot:call", "workflows:new"])
"""

from typing import Iterable
import logging

logger = logging.getLogger(__name__)

from .unified_event_bus import UnifiedEventBus
from .bot_registry import BotRegistry
from .neuroplasticity import PathwayDB, PathwayRecord, Outcome


class EventCollector:
    """Subscribe to bus topics and log relationships."""

    def __init__(
        self,
        event_bus: UnifiedEventBus,
        *,
        registry: BotRegistry | None = None,
        pathway_db: PathwayDB | None = None,
        topics: Iterable[str] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.registry = registry or BotRegistry()
        self.pathway_db = pathway_db
        self.topics = list(topics or ("bot:call", "code:new", "workflows:new", "memory:new"))
        for t in self.topics:
            try:
                self.event_bus.subscribe(t, self._on_event)
            except Exception:
                logger.exception("failed to subscribe to %s", t)

    # ------------------------------------------------------------------
    def _on_event(self, topic: str, payload: object) -> None:
        if topic == "bot:call" and isinstance(payload, dict):
            f = payload.get("from")
            t = payload.get("to")
            if f and t:
                try:
                    self.registry.register_interaction(str(f), str(t))
                except Exception:
                    logger.exception("failed registering interaction %s -> %s", f, t)
        if self.pathway_db:
            try:
                rec = PathwayRecord(
                    actions=topic,
                    inputs=str(payload),
                    outputs="",
                    exec_time=0.0,
                    resources="",
                    outcome=Outcome.SUCCESS,
                    roi=0.0,
                )
                self.pathway_db.log(rec)
            except Exception:
                logger.exception("failed logging pathway record")


__all__ = ["EventCollector"]
