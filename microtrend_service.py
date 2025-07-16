from __future__ import annotations

"""Service for continuous microtrend detection."""

import logging
import threading
import os
from typing import Callable, List, Sequence, Optional, TYPE_CHECKING

from .retry_utils import retry

from .trending_scraper import TrendingScraper, TrendingItem
from .knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .unified_event_bus import UnifiedEventBus
    from .menace_orchestrator import MenaceOrchestrator


class MicrotrendService:
    """Run :class:`TrendingScraper` and update the knowledge graph."""

    def __init__(
        self,
        scraper: TrendingScraper | None = None,
        knowledge: KnowledgeGraph | None = None,
        *,
        on_new: Callable[[Sequence[TrendingItem]], None] | None = None,
        planner: Callable[[Sequence[TrendingItem]], None] | None = None,
        orchestrator: "MenaceOrchestrator" | None = None,
        event_bus: "UnifiedEventBus" | None = None,
    ) -> None:
        self.scraper = scraper or TrendingScraper()
        self.knowledge = knowledge or KnowledgeGraph()
        self.logger = logging.getLogger("MicrotrendService")
        self.on_new = on_new
        self.planner = planner
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self._last_seen: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None

    # ------------------------------------------------------------------
    @retry(Exception, attempts=3, delay=0.1)
    def _send_hint(self, hint: str) -> None:
        """Publish *hint* via event bus or orchestrator."""
        if self.event_bus:
            self.event_bus.publish("scaling_hint", hint)
        elif self.orchestrator:
            self.orchestrator.receive_scaling_hint(hint)

    # ------------------------------------------------------------------
    def run_once(self) -> List[TrendingItem]:
        energy = None
        if hasattr(self.knowledge, "energy_score"):
            try:
                energy = self.knowledge.energy_score()
            except Exception as exc:
                energy = None
                self.logger.error("energy_score failed", exc_info=True)
        env_energy = os.getenv("MICROTREND_ENERGY")
        if env_energy is not None and energy is None:
            try:
                energy = float(env_energy)
            except ValueError as exc:
                energy = None
                self.logger.error("invalid env energy", exc_info=True)
        try:
            items = self.scraper.collect_all(energy)
        except TypeError:
            items = self.scraper.collect_all()
        micro = TrendingScraper.detect_microtrends(items)
        names = {it.product_name for it in micro if it.product_name}
        for name in names:
            try:
                self.knowledge.add_trending_item(name)
            except Exception:
                self.logger.error("failed adding trending item: %s", name, exc_info=True)
        new_names = names - self._last_seen
        if new_names:
            if self.on_new:
                try:
                    self.on_new(micro)
                except Exception:
                    self.logger.error("on_new callback failed", exc_info=True)
            if self.planner:
                try:
                    if callable(self.planner):
                        self.planner(micro)
                    elif hasattr(self.planner, "update"):
                        self.planner.update(micro)  # type: ignore[attr-defined]
                except Exception:
                    self.logger.error("planner failed", exc_info=True)
            hint = "scale_up"
            try:
                self._send_hint(hint)
            except Exception:
                self.logger.error("failed to send scaling hint", exc_info=True)
        self._last_seen = names
        return micro

    # ------------------------------------------------------------------
    def run_continuous(
        self, interval: float = 3600.0, *, stop_event: threading.Event | None = None
    ) -> threading.Thread:
        """Run :meth:`run_once` repeatedly in a thread."""

        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop = stop_event or threading.Event()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:
                    self.logger.error("run_once failed", exc_info=True)
                if self._stop.wait(interval):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self._thread


__all__ = ["MicrotrendService"]
