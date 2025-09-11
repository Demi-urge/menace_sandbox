"""Niche Saturation Bot for rapid domination of high potential niches."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None  # type: ignore
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .db_router import GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from .resource_allocation_bot import ResourceAllocationBot
from .resource_prediction_bot import ResourceMetrics
from .prediction_manager_bot import PredictionManager
from .strategy_prediction_bot import StrategyPredictionBot
from vector_service.context_builder import ContextBuilder
from snippet_compressor import compress_snippets


logger = logging.getLogger(__name__)


@dataclass
class NicheCandidate:
    """Potential niche with demand and competition indicators."""

    name: str
    demand: float
    competition: float
    trend: float = 0.0


@dataclass
class SaturationLog:
    """Record saturation actions and ROI."""

    niche: str
    roi: float
    ts: str = datetime.utcnow().isoformat()


class NicheDB:
    """SQLite storage for saturation history."""

    def __init__(self, path: str | Path = "niche_history.db") -> None:
        LOCAL_TABLES.add("saturation")
        p = Path(path).resolve()
        self.router = GLOBAL_ROUTER or init_db_router("niche_db", str(p), str(p))
        self.conn = self.router.get_connection("saturation")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saturation(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                niche TEXT,
                roi REAL,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, log: SaturationLog) -> int:
        cur = self.conn.execute(
            "INSERT INTO saturation(niche, roi, ts) VALUES (?, ?, ?)",
            (log.niche, log.roi, log.ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def history(self) -> pd.DataFrame:
        return pd.read_sql("SELECT niche, roi, ts FROM saturation", self.conn)


@self_coding_managed
class NicheSaturationBot:
    """Detect viable niches and coordinate rapid saturation."""

    prediction_profile = {"scope": ["niche"], "risk": ["medium"]}

    def __init__(
        self,
        db: NicheDB | None = None,
        alloc_bot: ResourceAllocationBot | None = None,
        *,
        prediction_manager: "PredictionManager" | None = None,
        strategy_bot: "StrategyPredictionBot" | None = None,
        context_builder: ContextBuilder,
    ) -> None:
        if context_builder is None:
            raise TypeError("context_builder is required")

        try:
            context_builder.refresh_db_weights()
        except Exception as exc:
            logger.exception("context builder refresh failed: %s", exc)
            raise RuntimeError("context builder refresh failed") from exc

        self.context_builder = context_builder
        self.db = db or NicheDB()
        self.alloc_bot = alloc_bot or ResourceAllocationBot(context_builder=self.context_builder)
        self.alloc_bot.context_builder = self.context_builder
        self.prediction_manager = prediction_manager
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.strategy_bot = strategy_bot
        if LogisticRegression is not None:
            self.model = LogisticRegression()
        else:  # pragma: no cover - fallback when sklearn missing
            self.model = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("NicheSaturationBot")

    def _apply_prediction_bots(self, base: float, cand: NicheCandidate) -> float:
        """Combine predictions from assigned bots for niche selection."""
        if not self.prediction_manager:
            return base
        score = base
        count = 1
        vec = [cand.demand, cand.competition, cand.trend]
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                val = pred(vec)
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                continue
        return float(score / count)

    def train(self, samples: Iterable[NicheCandidate], labels: Iterable[int]) -> None:
        """Train the niche viability model."""
        if self.model is None or np is None:
            return
        X = np.array([[s.demand, s.competition, s.trend] for s in samples])
        y = np.array(list(labels))
        self.model.fit(X, y)

    def detect(self, candidates: Iterable[NicheCandidate]) -> List[NicheCandidate]:
        """Return promising niches.

        The method now combines model predictions with a heuristic
        calculation.  If a scikit-learn model has been trained it will
        be used to obtain a probability of success which is then
        adjusted by any assigned prediction bots.  When a model is not
        available the heuristic calculation is still performed.
        """

        promising: List[NicheCandidate] = []
        for cand in candidates:
            try:
                ctx = self.context_builder.build(cand.name)
                ctx = compress_snippets({"snippet": ctx}).get("snippet", ctx)
            except Exception:
                ctx = ""

            use_model = (
                self.model is not None
                and np is not None
                and hasattr(self.model, "classes_")
            )

            if use_model:
                vec = [cand.demand, cand.competition, cand.trend]
                prob = float(self.model.predict_proba([vec])[0][1])
                score = self._apply_prediction_bots(prob, cand)
            else:
                score = (cand.demand * (1 + cand.trend)) / (cand.competition + 1)
                score = self._apply_prediction_bots(score, cand)

            if ctx:
                score *= 1 + min(len(ctx), 1000) / 10000.0

            if score > 0.5:
                promising.append(cand)

        return promising

    def saturate(self, candidates: Iterable[NicheCandidate]) -> List[Tuple[str, bool]]:
        """Allocate resources to promising niches and log ROI."""
        promising = self.detect(candidates)
        if self.strategy_bot:
            try:
                self.strategy_bot.receive_niche_info(promising)
            except Exception:
                self.logger.exception("strategy bot failed to receive niche info")
        metrics: Dict[str, ResourceMetrics] = {
            c.name: ResourceMetrics(cpu=1.0, memory=50.0, disk=1.0, time=1.0) for c in promising
        }
        actions = self.alloc_bot.allocate(metrics) if metrics else []
        for bot_name, active in actions:
            roi = 1.0 if active else 0.0
            self.db.add(SaturationLog(niche=bot_name, roi=roi))
        return actions


__all__ = [
    "NicheCandidate",
    "SaturationLog",
    "NicheDB",
    "NicheSaturationBot",
]
