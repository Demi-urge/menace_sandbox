"""Efficiency Bot for model optimisation and benchmarking."""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Type

from .bot_registry import BotRegistry
from .coding_bot_interface import self_coding_managed
from .data_bot import DataBot
from .sandbox_settings import SandboxSettings

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .db_router import LOCAL_TABLES
from .strategy_prediction_bot import StrategyPredictionBot
from .bot_database import BotDB
from .code_database import CodeDB
from .task_handoff_bot import WorkflowDB
from .unified_event_bus import UnifiedEventBus
from .contrarian_db import ContrarianDB
from .database_manager import DB_PATH


registry = BotRegistry()
_DATA_BOT: DataBot | None = None


def _bootstrap_fast_enabled() -> bool:
    """Return ``True`` when fast bootstrap mode should avoid heavy setup."""

    value = os.getenv("MENACE_BOOTSTRAP_FAST")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_data_bot() -> DataBot:
    """Return a shared :class:`DataBot`, creating it lazily when needed."""

    global _DATA_BOT
    if _DATA_BOT is not None:
        return _DATA_BOT

    bootstrap_fast = _bootstrap_fast_enabled()
    settings = None
    if bootstrap_fast:
        logger.info(
            "EfficiencyBot bootstrap_fast enabled; deferring heavy DataBot setup",
        )
        settings = SandboxSettings(bootstrap_fast=True, build_groups=False)

    _DATA_BOT = DataBot(
        start_server=False,
        settings=settings,
        bootstrap=bootstrap_fast,
    )
    return _DATA_BOT


_get_data_bot.__self_coding_lazy__ = True
data_bot = _get_data_bot

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from .capital_management_bot import CapitalManagementBot
    from .prediction_manager_bot import PredictionManager
    from .chatgpt_enhancement_bot import EnhancementDB
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when optional deps missing
    CapitalManagementBot = Any  # type: ignore[assignment]
    PredictionManager = Any  # type: ignore[assignment]
    EnhancementDB = Any  # type: ignore[assignment]
    SelfCodingManager = Any  # type: ignore[assignment]


@lru_cache(maxsize=1)
def _capital_management_bot_cls() -> Type["CapitalManagementBot"]:
    """Resolve :class:`CapitalManagementBot` lazily to avoid circular imports."""

    module = import_module(".capital_management_bot", __package__)
    return module.CapitalManagementBot  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _prediction_manager_cls() -> Type["PredictionManager"]:
    """Resolve :class:`PredictionManager` lazily to avoid circular imports."""

    module = import_module(".prediction_manager_bot", __package__)
    return module.PredictionManager  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _enhancement_db_cls() -> Type["EnhancementDB"]:
    """Resolve :class:`EnhancementDB` lazily to avoid circular imports."""

    module = import_module(".chatgpt_enhancement_bot", __package__)
    return module.EnhancementDB  # type: ignore[attr-defined]


@dataclass
class EfficiencyMetrics:
    """Model efficiency metrics."""

    latency: float
    throughput: float
    cost: float


class EfficiencyDB:
    """SQLite backed store for efficiency records with resilient connections."""

    def __init__(self, path: str | Path = "efficiency.db") -> None:
        LOCAL_TABLES.add("efficiency")
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection | None = None
        self._connect()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Establish a SQLite connection and ensure the schema exists."""

        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        logger.debug("[EfficiencyDB] Connected: %s", conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS efficiency(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT,
                latency REAL,
                throughput REAL,
                cost REAL,
                ts TEXT
            )
            """
        )
        conn.commit()
        self.conn = conn
        return conn

    # ------------------------------------------------------------------
    def _ensure_connection(self) -> sqlite3.Connection:
        """Return an active connection, reconnecting if the prior one was closed."""

        conn = self.conn
        if conn is None:
            return self._connect()

        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            if "closed" not in str(exc).lower():
                raise
            logger.warning("EfficiencyDB connection was closed; reconnecting.")
            return self._connect()

        return conn

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying SQLite connection."""

        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None

    # ------------------------------------------------------------------
    def add(self, model: str, metrics: EfficiencyMetrics) -> int:
        conn = self._ensure_connection()
        cur = conn.execute(
            "INSERT INTO efficiency(model, latency, throughput, cost, ts) VALUES (?,?,?,?,?)",
            (model, metrics.latency, metrics.throughput, metrics.cost, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    def history(self) -> List[Tuple[str, float, float, float, str]]:
        conn = self._ensure_connection()
        cur = conn.execute(
            "SELECT model, latency, throughput, cost, ts FROM efficiency ORDER BY id"
        )
        return cur.fetchall()


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class EfficiencyBot:
    """Optimise models via compression and benchmarking."""

    prediction_profile = {"scope": ["efficiency"], "risk": ["low"]}

    def __init__(
        self,
        db: EfficiencyDB | None = None,
        *,
        data_bot: "DataBot" | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        prediction_manager: "PredictionManager" | None = None,
        strategy_bot: "StrategyPredictionBot" | None = None,
        bot_db: Optional[BotDB] = None,
        code_db: Optional[CodeDB] = None,
        workflow_db: Optional[WorkflowDB] = None,
        contrarian_db: Optional[ContrarianDB] = None,
        enhancement_db: Optional["EnhancementDB"] = None,
        models_db: Path | str = DB_PATH,
        event_bus: UnifiedEventBus | None = None,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        self.db = db or EfficiencyDB()
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.prediction_manager = prediction_manager
        self.strategy_bot = strategy_bot
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.bot_db = bot_db or BotDB()
        self.code_db = code_db or CodeDB()
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        self.contrarian_db = contrarian_db or ContrarianDB()
        enhancement_db_cls = _enhancement_db_cls()
        self.enhancement_db = enhancement_db or enhancement_db_cls()
        self.models_db = Path(models_db)
        self.logger = logging.getLogger("EfficiencyBot")

    @staticmethod
    def compress_model(size: float) -> float:
        """Return a compressed size using simple quantisation and pruning."""
        return max(size * 0.5, size - 10.0)

    @staticmethod
    def benchmark(size: float) -> EfficiencyMetrics:
        """Benchmark a model by estimating latency and throughput."""
        latency = max(0.1, size / 100)
        throughput = 1000 / (latency * 1000)
        cost = latency * size * 0.01
        return EfficiencyMetrics(latency=latency, throughput=throughput, cost=cost)

    def _apply_prediction_bots(self, feats: Iterable[float]) -> float:
        """Combine predictions from assigned bots."""
        if not self.prediction_manager:
            return 0.0
        preds: List[float] = []
        for pid in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(pid)
            if not entry or not entry.bot:
                continue
            predict = getattr(entry.bot, "predict", None)
            if not callable(predict):
                continue
            try:
                preds.append(float(predict(list(feats))))
            except Exception:
                continue
        return float(sum(preds) / len(preds)) if preds else 0.0

    def optimise(self, models: Dict[str, float]) -> List[Tuple[str, EfficiencyMetrics]]:
        """Compress and benchmark models, logging results."""
        results: List[Tuple[str, EfficiencyMetrics]] = []
        for name, size in models.items():
            new_size = self.compress_model(size)
            metrics = self.benchmark(new_size)
            self.db.add(name, metrics)
            results.append((name, metrics))
            if self.data_bot:
                self.data_bot.collect(
                    bot=name,
                    response_time=metrics.latency,
                    revenue=0.0,
                    expense=metrics.cost,
                )
        return results

    def record_competitor(self, name: str, metrics: EfficiencyMetrics) -> None:
        """Store competitor data via DataBot for later analysis."""
        if self.data_bot:
            self.data_bot.collect(
                bot=name,
                response_time=metrics.latency,
                revenue=0.0,
                expense=metrics.cost,
            )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def current_metrics(self, limit: int = 50) -> pd.DataFrame:
        """Return recent performance metrics."""
        if not self.data_bot:
            return pd.DataFrame()
        try:
            return self.data_bot.db.fetch(limit)
        except Exception:
            return pd.DataFrame()

    def predict_micro_bottlenecks(self, features: Iterable[float]) -> float:
        """Combine predictions from assigned bots for hidden issues."""
        return self._apply_prediction_bots(features)

    def assess_efficiency(self) -> Dict[str, float]:
        """Evaluate current efficiency state and predict bottlenecks."""
        df = self.current_metrics(50)
        latency = float(df["response_time"].mean()) if not df.empty else 0.0
        errors = float(df["errors"].sum()) if not df.empty and "errors" in df.columns else 0.0
        roi = float(self.capital_bot.profit()) if self.capital_bot else 0.0
        base_features = [latency, errors, roi]
        bottleneck_score = self.predict_micro_bottlenecks(base_features)
        return {
            "latency": latency,
            "errors": errors,
            "roi": roi,
            "predicted_bottleneck": bottleneck_score,
        }

    def send_findings(self, report: Optional[Dict[str, float]] = None) -> None:
        """Transmit findings to the strategy prediction bot if available."""
        if not self.strategy_bot:
            return
        report = report or self.assess_efficiency()
        receiver = getattr(self.strategy_bot, "receive_efficiency_report", None)
        if callable(receiver):
            try:
                receiver(report)
            except Exception:
                self.logger.exception("failed to send efficiency report")


__all__ = [
    "EfficiencyMetrics",
    "EfficiencyDB",
    "EfficiencyBot",
]