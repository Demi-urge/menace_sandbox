"""Structural Evolution Bot predicts and logs system-wide architecture changes."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .data_bot import MetricsDB, DataBot
from .evolution_approval_policy import EvolutionApprovalPolicy
from .self_coding_manager import SelfCodingManager
from .bot_registry import BotRegistry

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

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


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class StructuralEvolutionBot:
    """Forecast and apply structural adjustments based on metrics."""

    def __init__(
        self,
        manager: SelfCodingManager | None = None,
        *,
        metrics_db: MetricsDB | None = None,
        db: EvolutionDB | None = None,
        approval_policy: "EvolutionApprovalPolicy | None" = None,
    ) -> None:
        self.manager = manager
        self.metrics_db = metrics_db or MetricsDB()
        self.db = db or EvolutionDB()
        self.approval_policy = approval_policy or EvolutionApprovalPolicy()
        if self.manager is not None:
            try:
                name = getattr(self, "name", getattr(self, "bot_name", self.__class__.__name__))
                self.manager.register_bot(name)
                orch = getattr(self.manager, "evolution_orchestrator", None)
                if orch:
                    orch.register_bot(name)
            except Exception:
                logger.exception("bot registration failed")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("StructuralEvolution")
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = data_bot

    def take_snapshot(self, limit: int = 100) -> SystemSnapshot:
        df = self.metrics_db.fetch(limit)
        return SystemSnapshot(metrics=df)

    def predict_changes(self, snap: SystemSnapshot) -> List[EvolutionRecord]:
        score = DataBot.complexity_score(snap.metrics)
        if score > 150:
            severity = "major"
            change = "redistribute_load"
        else:
            severity = "minor"
            change = "merge_idle_bots"
        impact = float(score * 0.1)
        rec = EvolutionRecord(change=change, impact=impact, severity=severity)
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
