"""Structural Evolution Bot predicts and logs system-wide architecture changes."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable, TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - ignore missing yaml
    yaml = None  # type: ignore

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .coding_bot_interface import (
    get_structural_bootstrap_owner as _get_structural_bootstrap_owner,
    normalise_manager_arg,
    prepare_pipeline_for_bootstrap,
    self_coding_managed,
    structural_bootstrap_owner_guard,
)
from .data_bot import MetricsDB, DataBot, persist_sc_thresholds
from .evolution_approval_policy import EvolutionApprovalPolicy
from .self_coding_manager import SelfCodingManager, internalize_coding_bot
from .bot_registry import BotRegistry
from .self_coding_engine import SelfCodingEngine
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
from vector_service.context_builder import ContextBuilder
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model_automation_pipeline import ModelAutomationPipeline
    from .evolution_orchestrator import EvolutionOrchestrator
else:  # pragma: no cover - runtime fallback avoids circular import on load
    ModelAutomationPipeline = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_registry: BotRegistry | None = None
_data_bot: DataBot | None = None
_context_builder: ContextBuilder | None = None
_engine: SelfCodingEngine | None = None
_pipeline: "ModelAutomationPipeline" | None = None
_pipeline_promoter: Callable[[object], None] | None = None
_manager: SelfCodingManager | None = None
_thresholds = None
_bootstrap_lock = threading.RLock()
_bootstrap_event = threading.Event()
_bootstrap_error: BaseException | None = None
_bootstrap_in_progress = False
# Legacy module-level handles populated after bootstrap for callers that still
# reference ``structural_evolution_bot.manager`` or ``.data_bot``.
registry: BotRegistry | None = None
data_bot: DataBot | None = None
manager: SelfCodingManager | None = None


def _get_registry() -> BotRegistry:
    global _registry
    if _registry is None:
        _registry = BotRegistry()
    return _registry


def _get_data_bot() -> DataBot:
    global _data_bot
    if _data_bot is None:
        _data_bot = DataBot(start_server=False)
    return _data_bot


def _get_context_builder() -> ContextBuilder:
    global _context_builder
    if _context_builder is None:
        _context_builder = create_context_builder()
    return _context_builder


def _get_engine() -> SelfCodingEngine:
    global _engine
    if _engine is None:
        _engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_get_context_builder())
    return _engine


def _load_thresholds():
    global _thresholds
    if _thresholds is None:
        _thresholds = get_thresholds("StructuralEvolutionBot")
        persist_sc_thresholds(
            "StructuralEvolutionBot",
            roi_drop=_thresholds.roi_drop,
            error_increase=_thresholds.error_increase,
            test_failure_increase=_thresholds.test_failure_increase,
        )
    return _thresholds


def _build_pipeline() -> tuple["ModelAutomationPipeline", Callable[[object], None]]:
    """Construct the automation pipeline without triggering circular imports."""

    from .model_automation_pipeline import ModelAutomationPipeline as _Pipeline

    return prepare_pipeline_for_bootstrap(
        pipeline_cls=_Pipeline,
        context_builder=_get_context_builder(),
        bot_registry=_get_registry(),
        data_bot=_get_data_bot(),
    )


def _prepare_or_wait_for_bootstrap(owner: object | None = None) -> SelfCodingManager:
    global _pipeline, _manager, _pipeline_promoter, _bootstrap_error, _bootstrap_in_progress

    with _bootstrap_lock:
        if _manager is not None:
            return _manager
        if _bootstrap_in_progress:
            logger.debug("StructuralEvolutionBot bootstrap already in progress; waiting for result")
            wait_event = True
        else:
            wait_event = False
            _bootstrap_in_progress = True
            _bootstrap_event.clear()

    if wait_event:
        if not _bootstrap_event.wait(timeout=30):
            raise TimeoutError("StructuralEvolutionBot bootstrap wait timed out")
        if _bootstrap_error is not None:
            raise _bootstrap_error
        if _manager is None:
            raise RuntimeError("StructuralEvolutionBot bootstrap did not produce a manager")
        return _manager

    owner_token = owner if owner is not None else object()
    with structural_bootstrap_owner_guard(owner_token):
        try:
            _pipeline, _pipeline_promoter = _build_pipeline()
            orchestrator = get_orchestrator("StructuralEvolutionBot", _get_data_bot(), _get_engine())
            th = _load_thresholds()
            _manager = internalize_coding_bot(
                "StructuralEvolutionBot",
                _get_engine(),
                _pipeline,
                data_bot=_get_data_bot(),
                bot_registry=_get_registry(),
                evolution_orchestrator=orchestrator,
                threshold_service=ThresholdService(),
                roi_threshold=th.roi_drop,
                error_threshold=th.error_increase,
                test_failure_threshold=th.test_failure_increase,
            )
            globals()["manager"] = _manager
            globals()["data_bot"] = _get_data_bot()
            globals()["registry"] = _get_registry()
            if _pipeline_promoter is not None:
                _pipeline_promoter(_manager)
            return _manager
        except BaseException as exc:  # pragma: no cover - propagate and record
            _bootstrap_error = exc
            raise
        finally:
            with _bootstrap_lock:
                _bootstrap_in_progress = False
                _bootstrap_event.set()


def get_structural_evolution_manager(owner: object | None = None) -> SelfCodingManager:
    """Return a bootstrapped :class:`SelfCodingManager` instance."""

    owner = owner or _get_structural_bootstrap_owner() or None
    return _prepare_or_wait_for_bootstrap(owner)


def active_structural_bootstrap_owner() -> object | None:
    """Expose the active bootstrap owner token, if any."""

    return _get_structural_bootstrap_owner()


_get_registry.__self_coding_lazy__ = True  # type: ignore[attr-defined]
_get_data_bot.__self_coding_lazy__ = True  # type: ignore[attr-defined]
get_structural_evolution_manager.__self_coding_lazy__ = True  # type: ignore[attr-defined]

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


@self_coding_managed(
    bot_registry=_get_registry,
    data_bot=_get_data_bot,
    manager=get_structural_evolution_manager,
)
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
        self.data_bot = _get_data_bot()
        fallback_manager = manager if manager is not None else get_structural_evolution_manager()
        self.manager = normalise_manager_arg(
            manager,
            type(self),
            fallback=fallback_manager,
        )

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
