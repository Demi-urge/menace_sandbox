"""Dynamic resource allocation framework using runtime metrics."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .data_bot import MetricsDB, MetricRecord
from .resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
from .resource_allocation_bot import ResourceAllocationBot, AllocationDB
from .neuroplasticity import PathwayDB
from .advanced_error_management import PredictiveResourceAllocator
from .resource_allocation_optimizer import ResourceAllocationOptimizer
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .menace_orchestrator import MenaceOrchestrator


@dataclass
class DecisionRecord:
    """Record of a resource allocation decision."""

    bot: str
    priority: float
    active: bool
    ts: str = datetime.utcnow().isoformat()


class DecisionLedger:
    """SQLite-backed ledger for resource decisions."""

    def __init__(self, path: Path | str = "decision_ledger.db") -> None:
        # allow connection access from multiple threads
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions(
                bot TEXT,
                priority REAL,
                active INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: DecisionRecord) -> None:
        self.conn.execute(
            "INSERT INTO decisions(bot, priority, active, ts) VALUES(?,?,?,?)",
            (rec.bot, rec.priority, int(rec.active), rec.ts),
        )
        self.conn.commit()

    def fetch(self) -> List[Tuple[str, float, bool, str]]:
        cur = self.conn.execute(
            "SELECT bot, priority, active, ts FROM decisions"
        )
        rows = cur.fetchall()
        return [(r[0], float(r[1]), bool(r[2]), r[3]) for r in rows]


class DynamicResourceAllocator:
    """Allocate resources dynamically based on metrics and predictions."""

    def __init__(
        self,
        metrics_db: MetricsDB | None = None,
        prediction_bot: ResourcePredictionBot | None = None,
        ledger: DecisionLedger | None = None,
        alloc_bot: ResourceAllocationBot | None = None,
        pathway_db: PathwayDB | None = None,
        predictive_allocator: PredictiveResourceAllocator | None = None,
        orchestrator: "MenaceOrchestrator" | None = None,
        optimizer: ResourceAllocationOptimizer | None = None,
        *,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
    ) -> None:
        self.metrics_db = metrics_db or MetricsDB()
        self.prediction_bot = prediction_bot or ResourcePredictionBot()
        self.ledger = ledger or DecisionLedger()
        self.alloc_bot = alloc_bot or ResourceAllocationBot(AllocationDB())
        self.pathway_db = pathway_db
        self.scaler = predictive_allocator or PredictiveResourceAllocator(self.metrics_db)
        self.orchestrator = orchestrator
        self.optimizer = optimizer
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DynamicAllocator")

    @staticmethod
    def _priority(metrics: ResourceMetrics) -> float:
        cost = metrics.cpu + metrics.memory / 100 + metrics.disk
        return 1.0 / (cost or 1.0)

    def _latest_metrics(self, bot: str) -> ResourceMetrics:
        df = self.metrics_db.fetch(50)
        df = df[df["bot"] == bot]
        if df.empty:
            return self.prediction_bot.predict(bot)
        row = df.iloc[0]
        return ResourceMetrics(
            cpu=float(row["cpu"]),
            memory=float(row["memory"]),
            disk=float(row["disk_io"]),
            time=float(row["response_time"]),
        )

    def allocate(
        self,
        bots: Iterable[str],
        *,
        weight: float = 1.0,
    ) -> List[Tuple[str, bool]]:
        metrics_map: Dict[str, ResourceMetrics] = {
            b: self._latest_metrics(b) for b in bots
        }
        weights = {b: weight for b in bots}
        boosted: Dict[str, float] = {}
        if self.pathway_db:
            try:
                top = self.pathway_db.top_pathways(len(bots))
                for pid, _ in top:
                    row = self.pathway_db.conn.execute(
                        "SELECT actions FROM pathways WHERE id=?",
                        (pid,),
                    ).fetchone()
                    if row and row[0].startswith("run_cycle:"):
                        name = row[0].split("run_cycle:", 1)[1].split("->")[0]
                        if name in weights:
                            weights[name] *= 1.5
                            boosted[name] = weights[name]
                for b in bots:
                    if b in boosted:
                        continue
                    row = self.pathway_db.conn.execute(
                        "SELECT id FROM pathways WHERE actions LIKE ? ORDER BY ts DESC LIMIT 1",
                        (f"run_cycle:{b}%",),
                    ).fetchone()
                    if row and self.pathway_db.is_highly_myelinated(int(row[0])):
                        weights[b] *= 1.5
                        boosted[b] = weights[b]
            except Exception:
                self.logger.error(
                    "Failed to adjust weights based on pathway DB", exc_info=True
                )
        actions = self.alloc_bot.allocate(metrics_map, weights=weights)
        final_actions: List[Tuple[str, bool]] = []
        for bot, active in actions:
            if bot.startswith("core_"):
                active = True
            factor = weights.get(bot, weight) / weight
            pr = self._priority(metrics_map[bot]) * factor
            self.ledger.add(DecisionRecord(bot=bot, priority=pr, active=active))
            final_actions.append((bot, active))
        self._maybe_scale()
        return final_actions

    def _maybe_scale(self) -> None:
        """Run predictive scaling using forecast and optimizer priorities."""
        try:
            self.scaler.forecast_and_allocate()
        except Exception:
            self.logger.error(
                "Failed to forecast and allocate resources", exc_info=True
            )
        try:
            df = self.metrics_db.fetch(10)
        except Exception:
            self.logger.error("Failed to fetch metrics", exc_info=True)
            return
        if df.empty:
            return
        mean_cpu = float(df["cpu"].mean()) / 100.0
        priority = 1.0
        if self.optimizer:
            try:
                weights = self.optimizer.bandit.weights
                if weights:
                    priority = max(float(w) for w in weights.values())
            except Exception:
                self.logger.error(
                    "Failed to read optimizer weights", exc_info=True
                )
        score = mean_cpu * priority
        hint = None
        if score > self.scale_up_threshold:
            hint = "scale_up"
        elif score < self.scale_down_threshold:
            hint = "scale_down"
        if hint and self.orchestrator:
            try:
                self.orchestrator.receive_scaling_hint(hint)
            except Exception:
                self.logger.error(
                    "Failed to send scaling hint to orchestrator", exc_info=True
                )


__all__ = ["DecisionRecord", "DecisionLedger", "DynamicResourceAllocator"]
