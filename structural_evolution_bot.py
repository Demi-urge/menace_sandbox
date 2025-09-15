"""Structural Evolution Bot predicts and logs system-wide architecture changes."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - ignore missing yaml
    yaml = None  # type: ignore

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .data_bot import MetricsDB, DataBot, persist_sc_thresholds
from .evolution_approval_policy import EvolutionApprovalPolicy
from .self_coding_manager import SelfCodingManager, internalize_coding_bot
from .bot_registry import BotRegistry
from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
from vector_service.context_builder import ContextBuilder
from typing import TYPE_CHECKING
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = create_context_builder()
engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_context_builder)
pipeline = ModelAutomationPipeline(context_builder=_context_builder)
evolution_orchestrator = get_orchestrator("StructuralEvolutionBot", data_bot, engine)
_th = get_thresholds("StructuralEvolutionBot")
persist_sc_thresholds(
    "StructuralEvolutionBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
    test_failure_increase=_th.test_failure_increase,
)
manager = internalize_coding_bot(
    "StructuralEvolutionBot",
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    evolution_orchestrator=evolution_orchestrator,
    threshold_service=ThresholdService(),
    roi_threshold=_th.roi_drop,
    error_threshold=_th.error_increase,
    test_failure_threshold=_th.test_failure_increase,
)

@dataclass
class SystemSnapshot:
    """Snapshot of system metrics at a point in time."""

    metrics: pd.DataFrame
    ts: str = datetime.utcnow().isoformat()


@dataclass
class EvolutionRecord:
    """Predicted structural change and its impact."""

    change: str
    impact: float
    severity: str
    status: str = "pending"
    ts: str = datetime.utcnow().isoformat()


class EvolutionDB:
    """SQLite-backed store for predicted evolutions."""

    def __init__(
        self,
        path: Path | str = "evolution.db",
        *,
        router: DBRouter | None = None,
    ) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "evolution_db", str(path), str(path)
        )
        self.conn = self.router.get_connection("evolutions")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evolutions(
                change TEXT,
                impact REAL,
                severity TEXT,
                status TEXT,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: EvolutionRecord) -> None:
        self.conn.execute(
            "INSERT INTO evolutions(change, impact, severity, status, ts) VALUES(?,?,?,?,?)",
            (rec.change, rec.impact, rec.severity, rec.status, rec.ts),
        )
        self.conn.commit()

    def fetch(self) -> List[Tuple[str, float, str, str, str]]:
        cur = self.conn.execute(
            "SELECT change, impact, severity, status, ts FROM evolutions"
        )
        return cur.fetchall()

    def update_status(self, change: str, status: str) -> None:
        self.conn.execute(
            "UPDATE evolutions SET status = ? WHERE change = ?",
            (status, change),
        )
        self.conn.commit()


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class StructuralEvolutionBot:
    """Forecast and apply structural adjustments based on metrics."""

    def __init__(
        self,
        *,
        metrics_db: MetricsDB | None = None,
        db: EvolutionDB | None = None,
        approval_policy: "EvolutionApprovalPolicy | None" = None,
        manager: SelfCodingManager | None = None,
    ) -> None:
        self.metrics_db = metrics_db or MetricsDB()
        self.db = db or EvolutionDB()
        self.approval_policy = approval_policy or EvolutionApprovalPolicy()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StructuralEvolution")
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = data_bot
        self.manager = manager or globals().get("manager")

    def take_snapshot(self, limit: int = 100) -> SystemSnapshot:
        df = self.metrics_db.fetch(limit)
        return SystemSnapshot(metrics=df)

    def _load_rules(self) -> List[Dict[str, object]]:
        path = Path(__file__).with_name("config").joinpath("complexity_mapping.yaml")
        if not path.exists() or yaml is None:
            return [
                {"multiplier": 1.0, "severity": "minor", "change": "merge_idle_bots"},
                {
                    "multiplier": 2.0,
                    "severity": "major",
                    "change": "redistribute_load",
                },
            ]
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or []
        if isinstance(data, dict):
            data = data.get("rules", [])
        return data  # type: ignore[return-value]

    def _dynamic_threshold(self, bot: str) -> float:
        base = 100.0
        service = getattr(self.manager, "threshold_service", None)
        if service is not None:
            try:
                th = service.get(self.name)
                base = float(getattr(th, "error_threshold", base))
            except Exception:
                base = 100.0
        try:
            avg_err = float(self.data_bot.average_errors(bot))
        except Exception:
            avg_err = 0.0
        return base + avg_err

    def _compute_impact(self, score: float, metrics: object) -> float:
        try:
            if pd is not None and hasattr(metrics, "empty") and not metrics.empty:
                series = metrics["cpu"] + metrics["memory"]
                deltas = series.diff().dropna()
                improvements = [-float(d) for d in deltas if d < 0]
                if improvements:
                    return float(sum(improvements) / len(improvements))
            elif isinstance(metrics, list) and len(metrics) > 1:
                vals = [r.get("cpu", 0.0) + r.get("memory", 0.0) for r in metrics]
                improvements = [max(vals[i] - vals[i + 1], 0.0) for i in range(len(vals) - 1)]
                if improvements:
                    return float(sum(improvements) / len(improvements))
        except Exception:
            pass
        return float(score * 0.1)

    def predict_changes(self, snap: SystemSnapshot) -> List[EvolutionRecord]:
        score = DataBot.complexity_score(snap.metrics)
        bot_name = self.name
        try:
            if pd is not None and hasattr(snap.metrics, "empty") and not snap.metrics.empty:
                bot_name = str(snap.metrics.iloc[-1]["bot"])
            elif isinstance(snap.metrics, list) and snap.metrics:
                bot_name = str(snap.metrics[-1].get("bot", self.name))
        except Exception:
            pass
        base_threshold = self._dynamic_threshold(bot_name)
        rules = self._load_rules()
        severity = str(rules[-1]["severity"])
        change = str(rules[-1]["change"])
        for rule in rules:
            limit = base_threshold * float(rule.get("multiplier", 1.0))
            if score <= limit:
                severity = str(rule["severity"])
                change = str(rule["change"])
                break
        impact = self._compute_impact(score, snap.metrics)
        rec = EvolutionRecord(change=change, impact=float(impact), severity=severity)
        self.db.add(rec)
        return [rec]

    def simulate(self, snap: SystemSnapshot, rec: EvolutionRecord) -> float:
        base = DataBot.complexity_score(snap.metrics)
        improved = max(base - rec.impact, 0.0)
        return improved

    def apply_minor_changes(self) -> List[str]:
        rows = self.db.fetch()
        applied: List[str] = []
        for change, impact, severity, status, _ in rows:
            if severity == "minor" and status == "pending":
                self.db.update_status(change, "applied")
                applied.append(change)
        return applied

    def apply_major_change(
        self, rec: EvolutionRecord, approve_cb: "Callable[[EvolutionRecord], bool] | None" = None
    ) -> bool:
        """Apply a major structural change using an approval policy."""
        if rec.severity != "major" or rec.status != "pending":
            return False
        approved = False
        if approve_cb:
            try:
                approved = bool(approve_cb(rec))
            except Exception:
                approved = False
        elif self.approval_policy:
            try:
                approved = bool(self.approval_policy.approve(rec))
            except Exception:
                approved = False
        self.db.update_status(rec.change, "applied" if approved else "denied")
        return approved


__all__ = [
    "SystemSnapshot",
    "EvolutionRecord",
    "EvolutionDB",
    "StructuralEvolutionBot",
    "EvolutionApprovalPolicy",
]
