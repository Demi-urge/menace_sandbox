"""Resource Allocation Optimizer with Bayesian bandit scheduling."""

from __future__ import annotations

import logging
import os
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING

from db_router import DBRouter, GLOBAL_ROUTER

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    from skopt import Optimizer
except Exception:  # pragma: no cover - optional dependency
    Optimizer = None  # type: ignore

from .failure_learning_system import DiscrepancyDB, FailureRecord, FailureLearningSystem
from .databases import MenaceDB
from .unified_event_bus import UnifiedEventBus
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry
from .performance_assessment_bot import SimpleRL
from .contextual_rl import ContextualRL
from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from .data_bot import MetricsDB
from .cross_query import workflow_roi_stats
from .retry_utils import retry

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from .error_bot import ErrorDB


@dataclass
class KPIRecord:
    """Record KPIs for a bot run."""

    bot: str
    revenue: float
    api_cost: float
    cpu_seconds: float
    success_rate: float
    ts: str = datetime.utcnow().isoformat()


class ROIDB:
    """SQLite store for per-run ROI metrics."""

    def __init__(self, path: Path | str = "roi.db", *, router: DBRouter = GLOBAL_ROUTER) -> None:
        self.path = str(Path(path))
        self.router = router
        conn = self.router.get_connection("roi")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roi(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot TEXT,
                revenue REAL,
                api_cost REAL,
                cpu_seconds REAL,
                success_rate REAL,
                ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_roi(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                revenue REAL,
                api_cost REAL,
                cpu_seconds REAL,
                success_rate REAL,
                ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS allocation_weights(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot TEXT,
                weight REAL,
                ts TEXT
            )
            """
        )
        conn.commit()

    def add(self, rec: KPIRecord) -> int:
        conn = self.router.get_connection("roi")
        cur = conn.execute(
            "INSERT INTO roi(bot, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
            (
                rec.bot,
                rec.revenue,
                rec.api_cost,
                rec.cpu_seconds,
                rec.success_rate,
                rec.ts,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def add_action_roi(
        self,
        action: str,
        revenue: float,
        api_cost: float,
        cpu_seconds: float,
        success_rate: float,
        ts: str | None = None,
    ) -> int:
        ts = ts or datetime.utcnow().isoformat()
        conn = self.router.get_connection("action_roi")
        cur = conn.execute(
            "INSERT INTO action_roi(action, revenue, api_cost, cpu_seconds, success_rate, ts) VALUES (?,?,?,?,?,?)",
            (
                action,
                revenue,
                api_cost,
                cpu_seconds,
                success_rate,
                ts,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def history(self, bot: str | None = None, limit: int = 50) -> pd.DataFrame:
        conn = self.router.get_connection("action_roi")
        if bot:
            query = (
                "SELECT action AS bot, revenue, api_cost, cpu_seconds, success_rate, ts FROM action_roi WHERE action=? ORDER BY id DESC LIMIT ?"
            )
            df = pd.read_sql(query, conn, params=(bot, limit))
            if df.empty:
                query = (
                    "SELECT bot, revenue, api_cost, cpu_seconds, success_rate, ts FROM roi WHERE bot=? ORDER BY id DESC LIMIT ?"
                )
                df = pd.read_sql(query, conn, params=(bot, limit))
            return df
        df1 = pd.read_sql(
            "SELECT action AS bot, revenue, api_cost, cpu_seconds, success_rate, ts FROM action_roi ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        df2 = pd.read_sql(
            "SELECT bot, revenue, api_cost, cpu_seconds, success_rate, ts FROM roi ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        return pd.concat([df1, df2], ignore_index=True)

    def add_weight(self, bot: str, weight: float, ts: str | None = None) -> int:
        ts = ts or datetime.utcnow().isoformat()
        conn = self.router.get_connection("allocation_weights")
        cur = conn.execute(
            "INSERT INTO allocation_weights(bot, weight, ts) VALUES (?,?,?)",
            (bot, weight, ts),
        )
        conn.commit()
        return int(cur.lastrowid)

    def weight_history(self, limit: int = 50) -> pd.DataFrame:
        conn = self.router.get_connection("allocation_weights")
        return pd.read_sql(
            "SELECT bot, weight, ts FROM allocation_weights ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )

    def future_roi(self, action: str, discount: float = 0.9) -> float:
        df = self.history(action, limit=5)
        if getattr(df, "empty", False) or len(df) < 2:
            return 0.0
        rois = [
            (row["revenue"] - row["api_cost"]) / (row["cpu_seconds"] or 1.0) * row["success_rate"]
            for _, row in df.iterrows()
        ]
        trend = rois[-1] - rois[0]
        return (rois[-1] + trend) * discount


class BayesianBandit:
    """Simple Bayesian bandit using scikit-optimize."""

    def __init__(self, bots: Iterable[str]) -> None:
        self.optimizers: Dict[str, Optimizer] = {}
        self.last_params: Dict[str, List[float]] = {}
        for b in bots:
            self.optimizers[b] = Optimizer([(0.0, 1.0)]) if Optimizer else None
        self.weights: Dict[str, float] = {b: 1.0 for b in bots}

    def update(self, scores: Dict[str, float]) -> Dict[str, float]:
        for bot, score in scores.items():
            opt = self.optimizers.get(bot)
            if not opt:
                self.weights[bot] = score
                continue
            last = self.last_params.get(bot)
            if last is not None:
                try:
                    opt.tell(last, -score)
                except Exception as exc:
                    self.logger.error("optimizer update failed for %s: %s", bot, exc)
            params = opt.ask()
            self.last_params[bot] = params
            self.weights[bot] = float(params[0])
        total = sum(self.weights.values()) or 1.0
        for bot in self.weights:
            self.weights[bot] /= total
        return dict(self.weights)


class ResourceAllocationOptimizer:
    """Optimise GPU time based on ROI and error metrics."""

    def __init__(
        self,
        roi_db: ROIDB | None = None,
        *,
        error_db: ErrorDB | None = None,
        menace_db: MenaceDB | None = None,
        discrepancy_db: DiscrepancyDB | None = None,
        grace_runs: int = 3,
        event_bus: Optional[UnifiedEventBus] = None,
        memory_mgr: MenaceMemoryManager | None = None,
        rl_model: SimpleRL | None = None,
        failure_system: FailureLearningSystem | None = None,
        evolution_history: EvolutionHistoryDB | None = None,
        async_mode: bool = False,
    ) -> None:
        self.roi_db = roi_db or ROIDB()
        self.error_db = error_db
        self.menace_db = menace_db
        self.discrepancy_db = discrepancy_db or DiscrepancyDB()
        self.grace_runs = grace_runs
        self.event_bus = event_bus
        self.memory_mgr = memory_mgr
        self.rl_model = rl_model or SimpleRL()
        self.failure_system = failure_system or FailureLearningSystem(self.discrepancy_db)
        self.evolution_history = evolution_history
        self.async_mode = async_mode
        self.last_error_event: object | None = None
        self.last_memory_entry: MemoryEntry | None = None
        self.fail_counts: Dict[str, int] = {}
        self.paused: Dict[str, bool] = {}
        self.workflow_fail_counts: Dict[str, int] = {}
        self.paused_workflows: Dict[str, bool] = {}
        self._workflow_event_ids: Dict[str, int] = {}
        self.bandit = BayesianBandit([])
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ResourceAllocationOptimizer")
        if self.event_bus:
            try:
                self.event_bus.subscribe("errors:new", self._on_error_event)
            except Exception as exc:
                self.logger.error("failed to subscribe to error events: %s", exc)
        if self.memory_mgr:
            try:
                self.memory_mgr.subscribe(self._on_memory_entry)
            except Exception as exc:
                self.logger.error("failed to subscribe to memory events: %s", exc)

    # ------------------------------------------------------------------
    def record_run(self, rec: KPIRecord) -> None:
        self.roi_db.add(rec)
        try:
            self.roi_db.add_action_roi(
                rec.bot,
                rec.revenue,
                rec.api_cost,
                rec.cpu_seconds,
                rec.success_rate,
                rec.ts,
            )
        except Exception as exc:
            self.logger.error("failed to add action ROI: %s", exc)

    def _error_cost(self, bot: str) -> float:
        if not self.error_db:
            return 0.0
        try:
            df = self.error_db.discrepancies()
            return float(len(df[df["message"].str.contains(bot)])) * 0.1
        except Exception as exc:
            self.logger.error("failed to compute error cost for %s: %s", bot, exc)
            return 0.0

    def _roi(self, bot: str) -> float:
        df = self.roi_db.history(bot, limit=5)
        if df.empty:
            return 0.0
        rev = float(df["revenue"].mean())
        cost = float(df["api_cost"].mean()) + self._error_cost(bot)
        cpu = float(df["cpu_seconds"].mean()) or 1.0
        succ = float(df["success_rate"].mean())
        return (rev - cost) / cpu * succ

    def _roi_trend(self, bot: str) -> float:
        df = self.roi_db.history(bot, limit=5)
        if df.empty or len(df) < 2:
            return 0.0
        rois = [
            (row["revenue"] - row["api_cost"]) / (row["cpu_seconds"] or 1.0) * row["success_rate"]
            for _, row in df.iterrows()
        ]
        return rois[-1] - rois[0]

    def _energy_score(self, bot: str) -> float:
        import math

        roi = self._roi(bot)
        trend = self._roi_trend(bot)
        base = roi + trend
        try:
            return 1.0 / (1.0 + math.exp(-base))
        except Exception:
            self.logger.error("energy score overflow for %s", bot)
            return max(0.0, min(1.0, base))

    def update_priorities(
        self,
        bots: Iterable[str],
        *,
        workflows: Iterable[str] | None = None,
        metrics_db: MetricsDB | None = None,
        prune_threshold: float = 0.0,
    ) -> Dict[str, float]:
        scores = {}
        for b in bots:
            roi = self._roi(b)
            trend = self._roi_trend(b)
            energy = self._energy_score(b)
            fail_penalty = self.failure_system.failure_score(b) if self.failure_system else 0.0
            reward = roi - fail_penalty
            state = (
                round(trend, 2),
                float(self.fail_counts.get(b, 0)),
                round(fail_penalty, 2),
                round(energy, 2),
            )
            self.rl_model.update(state, reward)
            scores[b] = self.rl_model.score(state)
        if self.evolution_history:
            try:
                delta = float(self.evolution_history.summary(10)["avg_delta"])
            except Exception as exc:
                self.logger.error("failed to compute evolution delta: %s", exc)
                delta = 0.0
            for b in scores:
                scores[b] *= 1.0 + delta
        self.bandit = BayesianBandit(bots)
        weights = self.bandit.update(scores)
        for bot, wt in weights.items():
            try:
                self.roi_db.add_weight(bot, float(wt))
            except Exception as exc:
                self.logger.error("failed to add weight for %s: %s", bot, exc)
        for bot, score in scores.items():
            if score <= 0.0:
                self.fail_counts[bot] = self.fail_counts.get(bot, 0) + 1
                if self.fail_counts[bot] > self.grace_runs:
                    self.paused[bot] = True
                    if self.discrepancy_db:
                        try:
                            self.discrepancy_db.log(
                                FailureRecord(
                                    model_id=bot,
                                    cause="low ROI",
                                    features="",
                                    demographics="",
                                    profitability=0.0,
                                    retention=0.0,
                                    cac=0.0,
                                    roi=score,
                                )
                            )
                        except Exception as exc:
                            self.logger.error("failed to log discrepancy: %s", exc)
            else:
                self.fail_counts[bot] = 0
        if workflows and metrics_db:
            try:
                self.prune_workflows(workflows, metrics_db=metrics_db, threshold=prune_threshold)
            except Exception as exc:
                self.logger.error("failed to prune workflows: %s", exc)
        return weights

    # ------------------------------------------------------------------
    def available_workflows(self) -> List[str]:
        """Return workflow names from ``MenaceDB`` if available."""
        if not self.menace_db:
            return []
        try:
            with self.menace_db.engine.connect() as conn:
                rows = conn.execute(self.menace_db.workflows.select()).mappings().fetchall()
            return [str(r["workflow_name"]) for r in rows]
        except Exception as exc:
            self.logger.error("failed to load workflows: %s", exc)
            return []

    def _disable_workflow(self, name: str) -> None:
        """Mark the workflow as paused and publish an event."""
        if self.menace_db:
            try:
                with self.menace_db.engine.begin() as conn:
                    row = (
                        conn.execute(
                            self.menace_db.workflows.select().where(
                                self.menace_db.workflows.c.workflow_name == name
                            )
                        )
                        .mappings()
                        .fetchone()
                    )
                    if row:
                        conn.execute(
                            self.menace_db.workflows.update()
                            .where(self.menace_db.workflows.c.workflow_id == row["workflow_id"])
                            .values(status="paused")
                        )
            except Exception as exc:
                self.logger.error("failed to mark workflow '%s' paused: %s", name, exc)
        if self.event_bus:
            try:
                self.event_bus.publish("workflow:disabled", {"workflow": name})
            except Exception as exc:
                self.logger.error("failed to publish workflow:disabled for %s: %s", name, exc)

    def prune_workflows(
        self,
        workflows: Iterable[str],
        *,
        metrics_db: MetricsDB,
        threshold: float = 0.0,
    ) -> List[str]:
        """Disable workflows whose ROI stays below ``threshold``."""
        removed: List[str] = []
        for wf in workflows:
            stats = workflow_roi_stats(wf, self.roi_db, metrics_db)
            roi = stats["roi"]
            if roi <= threshold:
                self.workflow_fail_counts[wf] = self.workflow_fail_counts.get(wf, 0) + 1
                if self.workflow_fail_counts[wf] > self.grace_runs:
                    if not self.paused_workflows.get(wf):
                        self.paused_workflows[wf] = True
                        self._disable_workflow(wf)
                        removed.append(wf)
            else:
                self.workflow_fail_counts[wf] = 0
        if removed and self.evolution_history:
            for wf in removed:
                try:
                    roi = workflow_roi_stats(wf, self.roi_db, metrics_db)["roi"]
                    parent = self._workflow_event_ids.get(wf)
                    change = 0.0 - roi
                    event_id = self.evolution_history.add(
                        EvolutionEvent(
                            action=f"prune:{wf}",
                            before_metric=roi,
                            after_metric=0.0,
                            roi=roi,
                            reason="prune low ROI workflow",
                            trigger="roi_threshold",
                            performance=change,
                            parent_event_id=parent,
                        )
                    )
                    self._workflow_event_ids[wf] = event_id
                except Exception as exc:
                    self.logger.error("failed to log evolution event: %s", exc)
        return removed

    # ------------------------------------------------------------------
    def _validate_response(self, resp: object) -> bool:
        try:
            status = resp.status_code
            data = resp.json()
        except Exception:
            self.logger.error("invalid autoscaler response")
            return False
        return status == 200 and data.get("status") == "ok"

    def _autoscale_sync(self, action: str, model_id: str, amount: int) -> bool:
        endpoint = os.getenv("AUTOSCALER_ENDPOINT")
        if not endpoint or not requests:
            return False
        url = f"{endpoint}/{action}"

        @retry(Exception, attempts=3)
        def _post() -> object:
            return requests.post(
                url,
                json={"model_id": model_id, "amount": amount},
                timeout=5,
            )

        try:
            resp = _post()
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("Autoscaler error: %s", exc)
            return False
        return self._validate_response(resp)

    async def _autoscale_async(self, action: str, model_id: str, amount: int) -> bool:
        endpoint = os.getenv("AUTOSCALER_ENDPOINT")
        if not endpoint or not requests:
            return False
        url = f"{endpoint}/{action}"

        @retry(Exception, attempts=3)
        def _post_sync() -> object:
            return requests.post(
                url,
                json={"model_id": model_id, "amount": amount},
                timeout=5,
            )

        try:
            resp = await asyncio.to_thread(_post_sync)
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("Autoscaler error: %s", exc)
            return False
        return self._validate_response(resp)

    def scale_up(self, model_id: str, amount: int = 1) -> bool:
        success = (
            asyncio.run(self._autoscale_async("scale_up", model_id, amount))
            if self.async_mode
            else self._autoscale_sync("scale_up", model_id, amount)
        )
        self.logger.info("scale up %s by %s", model_id, amount)
        return success

    async def scale_up_async(self, model_id: str, amount: int = 1) -> bool:
        success = await self._autoscale_async("scale_up", model_id, amount)
        self.logger.info("scale up %s by %s", model_id, amount)
        return success

    def scale_down(self, model_id: str, amount: int = 1) -> bool:
        success = (
            asyncio.run(self._autoscale_async("scale_down", model_id, amount))
            if self.async_mode
            else self._autoscale_sync("scale_down", model_id, amount)
        )
        self.logger.info("scale down %s by %s", model_id, amount)
        return success

    async def scale_down_async(self, model_id: str, amount: int = 1) -> bool:
        success = await self._autoscale_async("scale_down", model_id, amount)
        self.logger.info("scale down %s by %s", model_id, amount)
        return success

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_error_event(self, topic: str, payload: object) -> None:
        self.last_error_event = payload

    def _on_memory_entry(self, entry: MemoryEntry) -> None:
        if "roi" in (entry.tags or "").lower():
            self.last_memory_entry = entry


__all__ = [
    "KPIRecord",
    "ROIDB",
    "BayesianBandit",
    "ResourceAllocationOptimizer",
    "ContextualRL",
]
