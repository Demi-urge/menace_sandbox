"""Resources Bot for strategic allocation across Menace."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from vector_service import EmbeddableDBMixin
from resource_vectorizer import ResourceVectorizer
from .resource_allocation_bot import ResourceAllocationBot, AllocationDB
from .resource_prediction_bot import ResourceMetrics
from vector_service.context_builder import ContextBuilder
from .prediction_manager_bot import PredictionManager
from .strategy_prediction_bot import StrategyPredictionBot


@dataclass
class ROIRecord:
    """Record of ROI for a bot at a specific time."""

    bot: str
    roi: float
    ts: str = datetime.utcnow().isoformat()


class ROIHistoryDB(EmbeddableDBMixin):
    """SQLite-backed ROI history storage with embedding support."""

    def __init__(
        self,
        path: str | Path = "roi_history.db",
        *,
        router: DBRouter | None = None,
        vector_index_path: str = "resource_embeddings.index",
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "roi_history", str(path), str(path)
        )
        self.conn = self.router.get_connection("roi")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roi(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot TEXT,
                roi REAL,
                ts TEXT
            )
            """
        )
        self.conn.commit()
        meta_path = Path(vector_index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    def add(self, rec: ROIRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO roi(bot, roi, ts) VALUES (?, ?, ?)",
            (rec.bot, rec.roi, rec.ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def history(self, bot: str | None = None) -> pd.DataFrame:
        if bot:
            return pd.read_sql("SELECT bot, roi, ts FROM roi WHERE bot=?", self.conn, params=(bot,))
        return pd.read_sql("SELECT bot, roi, ts FROM roi", self.conn)

    # Embedding mixin hooks -------------------------------------------------
    def iter_records(self) -> Iterable[Tuple[int, Dict[str, float], str]]:
        cur = self.conn.execute("SELECT id, bot, roi, ts FROM roi")
        for row in cur.fetchall():
            rec = {"bot": row["bot"], "roi": row["roi"], "ts": row["ts"]}
            yield row["id"], rec, "resource"

    def vector(self, rec: Dict[str, float]) -> List[float]:
        return ResourceVectorizer().transform(rec)


class ResourcesBot:
    """Central allocator leveraging prediction data and ROI trends."""

    prediction_profile = {"scope": ["resources"], "risk": ["low"]}

    def __init__(
        self,
        db: ROIHistoryDB | None = None,
        alloc_bot: ResourceAllocationBot | None = None,
        *,
        context_builder: ContextBuilder,
        prediction_manager: "PredictionManager" | None = None,
        strategy_bot: "StrategyPredictionBot" | None = None,
    ) -> None:
        self.db = db or ROIHistoryDB()
        self.alloc_bot = alloc_bot or ResourceAllocationBot(
            AllocationDB(), context_builder=context_builder
        )
        self.strategy_bot = strategy_bot
        self.prediction_manager = prediction_manager
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ResourcesBot")

    def _apply_prediction_bots(self, base: float, metrics: ResourceMetrics) -> float:
        """Combine predictions from assigned bots."""
        if not self.prediction_manager:
            return base
        score = base
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            predict = getattr(entry.bot, "predict", None)
            if not callable(predict):
                continue
            try:
                other = predict([metrics.cpu, metrics.memory, metrics.disk, metrics.time])
                if isinstance(other, (list, tuple)):
                    other = other[0]
                score = (score + float(other)) / 2.0
            except Exception:
                continue
        return float(score)

    def assess_roi(
        self, metrics: Dict[str, ResourceMetrics], external: Dict[str, float] | None = None
    ) -> Dict[str, float]:
        """Compute simple ROI scores influenced by external factors."""
        external = external or {}
        factor = external.get("market", 1.0)
        scores: Dict[str, float] = {}
        for bot, m in metrics.items():
            base = 1.0 / (m.cpu + m.memory / 100 + m.disk + 1e-6)
            score = base * factor
            score = self._apply_prediction_bots(score, m)
            scores[bot] = score
        return scores

    def redistribute(
        self, metrics: Dict[str, ResourceMetrics], external: Dict[str, float] | None = None
    ) -> List[Tuple[str, bool]]:
        """Record ROI and delegate to the allocation bot for action."""
        scores = self.assess_roi(metrics, external)
        for bot, score in scores.items():
            self.db.add(ROIRecord(bot=bot, roi=score))
        actions = self.alloc_bot.allocate(metrics)
        if self.strategy_bot:
            try:
                self.strategy_bot.receive_resource_usage(metrics)
            except Exception:
                bot_name = getattr(
                    self.strategy_bot, "name", self.strategy_bot.__class__.__name__
                )
                self.logger.exception(
                    "strategy bot %s failed to receive resource usage", bot_name
                )
        return actions

    def current_allocations(self) -> pd.DataFrame:
        """Return historical allocation records."""
        return self.alloc_bot.db.history()


__all__ = ["ROIRecord", "ROIHistoryDB", "ResourcesBot"]
