from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import List, Optional
import logging

from .trending_scraper import TrendingScraper, TrendingItem
from .menace_discovery_engine import run_cycle as discovery_run_cycle
from .research_aggregator_bot import InfoDB, ResearchItem
from .bot_creation_bot import BotCreationBot
from .bot_planning_bot import PlanningTask
from .normalize_scraped_data import load_items, normalize
from vector_service import ContextBuilder


class DiscoveryScheduler:
    """Periodically run discovery and bot creation workflows."""

    def __init__(
        self,
        *,
        scraper: TrendingScraper | None = None,
        creation_bot: BotCreationBot | None = None,
        context_builder: ContextBuilder,
        interval: int = 3600,
    ) -> None:
        self.scraper = scraper or TrendingScraper()
        self.context_builder = context_builder
        try:
            self.context_builder.refresh_db_weights()
        except Exception:
            pass
        self.creation_bot = creation_bot or BotCreationBot(
            context_builder=self.context_builder
        )
        self.interval = interval
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("DiscoveryScheduler")

    # ------------------------------------------------------------------
    def _save_items(self, db: InfoDB, items: List[TrendingItem]) -> None:
        for it in items:
            try:
                db.add(
                    ResearchItem(
                        topic=it.product_name or it.platform,
                        content=it.source_url or "",
                        timestamp=time.time(),
                        title=it.product_name or it.platform,
                        tags=it.tags,
                        category=it.niche or "",
                        type_="trending",
                        source_url=it.source_url or "",
                    )
                )
            except Exception:
                self.logger.exception("failed to save item %s", it)

    def _new_candidates(self) -> List[str]:
        path = Path("niche_candidates.json")
        if not path.exists():
            return []
        data = load_items([path])
        candidates = normalize(data)
        return [c.product_name for c in candidates if c.product_name]

    def _create_bots(self, names: List[str]) -> None:
        tasks = [
            PlanningTask(
                description=name,
                complexity=1,
                frequency=1,
                expected_time=0.1,
                actions=[name.replace(" ", "_")],
            )
            for name in names
        ]
        if tasks:
            asyncio.run(self.creation_bot.create_bots(tasks))

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        db = InfoDB()
        while self.running:
            try:
                items = self.scraper.scrape_reddit()
                self._save_items(db, items)
            except Exception:
                self.logger.exception("scraper failed")
            try:
                asyncio.run(discovery_run_cycle())
            except Exception:
                self.logger.exception("discovery run failed")
            try:
                self._create_bots(self._new_candidates())
            except Exception:
                self.logger.exception("bot creation failed")
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None


__all__ = ["DiscoveryScheduler"]
