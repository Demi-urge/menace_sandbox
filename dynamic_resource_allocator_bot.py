"""Dynamic resource allocation framework using runtime metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope

from .data_bot import MetricsDB
from .resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
from .resource_allocation_bot import ResourceAllocationBot, AllocationDB
from .neuroplasticity import PathwayDB
from .advanced_error_management import PredictiveResourceAllocator
from .resource_allocation_optimizer import ResourceAllocationOptimizer
from vector_service.context_builder import ContextBuilder
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

    def __init__(
        self,
        path: Path | str = "decision_ledger.db",
        router: DBRouter | None = None,
    ) -> None:
        # allow connection access from multiple threads
        LOCAL_TABLES.add("decisions")
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "dynamic_resource_allocator", local_db_path=str(path), shared_db_path=str(path)
        )
        self.conn = self.router.get_connection("decisions")
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

    def fetch(
        self, *, scope: Scope | str = "local"
    ) -> List[Tuple[str, float, bool, str]]:
        """Return decision records filtered by menace ``scope``."""

        base = "SELECT bot, priority, active, ts FROM decisions"
        clause, params = build_scope_clause("decisions", scope, self.router.menace_id)
        base = apply_scope(base, clause)
        cur = self.conn.execute(base, params)
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
        context_builder: ContextBuilder | None = None,  # nocb
    ) -> None:
        self.metrics_db = metrics_db or MetricsDB()
        self.prediction_bot = prediction_bot or ResourcePredictionBot()
        self.ledger = ledger or DecisionLedger()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DynamicAllocator")

        self.context_builder = context_builder
        if alloc_bot is None:
            self.context_builder = self.context_builder or ContextBuilder()
            try:
                self.context_builder.refresh_db_weights()
            except Exception as exc:  # pragma: no cover - log then raise
                self.logger.error("context builder refresh failed: %s", exc)
                raise RuntimeError("context builder refresh failed") from exc
            self.alloc_bot = ResourceAllocationBot(
                AllocationDB(), context_builder=self.context_builder
            )
        else:
            self.alloc_bot = alloc_bot
            if self.context_builder is not None:
                try:
                    self.context_builder.refresh_db_weights()
                except Exception as exc:  # pragma: no cover - log then raise
                    self.logger.error("context builder refresh failed: %s", exc)
                    raise RuntimeError("context builder refresh failed") from exc

        self.pathway_db = pathway_db
        self.scaler = predictive_allocator or PredictiveResourceAllocator(self.metrics_db)
        self.orchestrator = orchestrator
        self.optimizer = optimizer
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

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
