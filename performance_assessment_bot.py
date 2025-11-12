"""Performance Assessment Bot evaluating KPIs and advising upgrades."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple
import logging

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    logging.getLogger(__name__).warning(
        "pandas is not installed; install with 'pip install pandas' to enable DataFrame support"
    )

from . import data_bot as db
from .bot_performance_history_db import BotPerformanceHistoryDB, PerformanceRecord


logger = logging.getLogger(__name__)


@dataclass
class KPI:
    """Key performance indicators used for projections."""

    cpu: float
    memory: float
    response_time: float
    errors: int


class SimpleRL:
    """Very small reinforcement learning helper using value updates."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.values: Dict[Tuple[int, int, int, int], float] = {}

    def update(self, state: Tuple[int, int, int, int], reward: float) -> float:
        q = self.values.get(state, 0.0)
        q += self.alpha * (reward - q)
        self.values[state] = q
        return q

    def score(self, state: Tuple[int, int, int, int]) -> float:
        return self.values.get(state, 0.0)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class PerformanceAssessmentBot:
    """Assess performance data and suggest hardware or software upgrades."""

    def __init__(
        self,
        metrics_db: db.MetricsDB | None = None,
        model: SimpleRL | None = None,
        history_db: BotPerformanceHistoryDB | None = None,
        *,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        self.db = metrics_db or db.MetricsDB()
        self.model = model or SimpleRL()
        self.history_db = history_db or BotPerformanceHistoryDB()
        self.history: Dict[str, float] = {}
        self.load_history()

    # ------------------------------------------------------------------
    def load_history(self) -> None:
        """Pre-populate RL model from stored performance history."""
        try:
            cur = self.history_db.conn.execute(
                "SELECT DISTINCT bot FROM bot_performance"
            )
            bots = [r[0] for r in cur.fetchall()]
        except Exception:
            bots = []
        for bot in bots:
            try:
                rows = self.history_db.history(bot, limit=50)
            except Exception:
                continue
            for row in rows:
                cpu, memory, response_time, errors = row[:4]
                state = (
                    int(cpu // 10),
                    int(memory // 10),
                    int(response_time // 1),
                    int(errors),
                )
                reward = -(cpu + memory + errors * 10)
                score = self.model.update(state, reward)
                self.history[bot] = score

    @staticmethod
    def _state_from_row(row: Mapping[str, Any]) -> Tuple[int, int, int, int]:
        return (
            int(row["cpu"] // 10),
            int(row["memory"] // 10),
            int(row["response_time"] // 1),
            int(row["errors"]),
        )

    def self_assess(self, bot_name: str, limit: int = 10) -> float:
        df = self.db.fetch(limit)
        if pd is None:
            rows = [r for r in df if r.get("bot") == bot_name]
            if not rows:
                return 0.0
            row = rows[0]
        else:
            df = df[df["bot"] == bot_name]
            if df.empty:
                return 0.0
            row = df.iloc[0]
        state = self._state_from_row(row)
        reward = -(row["cpu"] + row["memory"] + row["errors"] * 10)
        score = self.model.update(state, reward)
        self.history[bot_name] = score
        try:
            roi = float(row.get("revenue", 0.0) - row.get("expense", 0.0))
        except Exception:
            roi = 0.0
        try:
            rec = PerformanceRecord(
                bot=bot_name,
                cpu=float(row["cpu"]),
                memory=float(row["memory"]),
                response_time=float(row["response_time"]),
                errors=int(row["errors"]),
                roi=roi,
                score=score,
                disk_io=float(row.get("disk_io", 0.0)),
                net_io=float(row.get("net_io", 0.0)),
                revenue=float(row.get("revenue", 0.0)),
                expense=float(row.get("expense", 0.0)),
            )
            self.history_db.add(rec)
        except Exception as exc:
            logger.exception(
                "Failed to add performance record for %s: %s", bot_name, exc
            )
        return score

    def hypothetical_projection(self, kpi: KPI) -> float:
        base = 100.0 - (kpi.cpu + kpi.memory)
        base -= kpi.errors * 5
        return base - kpi.response_time * 10

    def suggest_enhancement(self, bot_name: str) -> str:
        score = self.history.get(bot_name, 0.0)
        if score < -50:
            return "Upgrade hardware"
        if score < -10:
            return "Optimise software"
        return "No action"

    def advise(self, bot_name: str) -> str:
        suggestion = self.suggest_enhancement(bot_name)
        return f"{bot_name} status: {suggestion}"


__all__ = [
    "KPI",
    "SimpleRL",
    "PerformanceAssessmentBot",
    "BotPerformanceHistoryDB",
    "PerformanceRecord",
]
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when manager is unused
    SelfCodingManager = object  # type: ignore[assignment]