"""Diagnostics and self-healing manager."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from db_router import GLOBAL_ROUTER, LOCAL_TABLES, init_db_router

try:  # pragma: no cover - support package and flat layouts
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .data_bot import MetricsDB
from .error_bot import ErrorBot, ErrorDB
from .coordination_manager import CoordinationManager
from .dynamic_resource_allocator_bot import DynamicResourceAllocator, DecisionLedger


@dataclass
class ResolutionRecord:
    """Entry describing a resolution action."""

    issue: str
    action: str
    success: bool
    ts: str = datetime.utcnow().isoformat()


class ResolutionDB:
    """SQLite store for resolution logs."""

    def __init__(self, path: Path | str = "resolutions.db") -> None:
        LOCAL_TABLES.add("resolutions")
        p = Path(path).resolve()
        self.router = GLOBAL_ROUTER or init_db_router(
            "resolutions_db", str(p), str(p)
        )
        self.conn = self.router.get_connection("resolutions")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS resolutions(
                issue TEXT,
                action TEXT,
                success INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: ResolutionRecord) -> None:
        self.conn.execute(
            "INSERT INTO resolutions(issue, action, success, ts) VALUES(?,?,?,?)",
            (rec.issue, rec.action, int(rec.success), rec.ts),
        )
        self.conn.commit()

    def fetch(self) -> List[Tuple[str, str, bool, str]]:
        cur = self.conn.execute(
            "SELECT issue, action, success, ts FROM resolutions"
        )
        rows = cur.fetchall()
        return [(r[0], r[1], bool(r[2]), r[3]) for r in rows]


class DiagnosticManager:
    """Diagnose inefficiencies and perform automated resolutions."""

    def __init__(
        self,
        metrics_db: MetricsDB | None = None,
        error_bot: ErrorBot | None = None,
        *,
        context_builder: "ContextBuilder" | None = None,
        ledger: DecisionLedger | None = None,
        queue: CoordinationManager | None = None,
        log: ResolutionDB | None = None,
    ) -> None:
        self.metrics = metrics_db or MetricsDB()
        self.context_builder = context_builder
        if error_bot is None:
            if self.context_builder is None:
                from vector_service.context_builder import ContextBuilder

                self.context_builder = ContextBuilder()
            self.error_bot = ErrorBot(
                ErrorDB(),
                self.metrics,
                context_builder=self.context_builder,
            )
        else:
            self.error_bot = error_bot
        self.ledger = ledger or DecisionLedger()
        self.queue = queue or CoordinationManager()
        self.log = log or ResolutionDB()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DiagnosticManager")

    def diagnose(self, limit: int = 20) -> List[str]:
        """Return list of issue descriptions based on metrics and discrepancies."""
        df = self.metrics.fetch(limit)
        issues: List[str] = []
        if not df.empty:
            mean_resp = df["response_time"].mean()
            if mean_resp > 2.0:
                issues.append("high_response_time")
            if df["errors"].sum() > 0:
                issues.append("error_rate")
        disc = self.error_bot.db.discrepancies()
        if not disc.empty:
            issues.append("discrepancies_detected")
        return issues

    def _restart_bot(self, bot: str) -> bool:
        self.logger.info("Restarting bot %s", bot)
        try:
            import subprocess
            subprocess.run(["pkill", "-f", bot], check=False)
            subprocess.Popen(
                ["python", str(resolve_path(f"{bot}.py"))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:  # pragma: no cover - runtime restart errors
            self.logger.error("Bot restart failed: %s", exc)
            return False

    def _reroute_tasks(self) -> bool:
        self.logger.info("Rerouting pending tasks")
        msg = self.queue.receive()
        if msg:
            self.queue.send(msg)
        return True

    def resolve_issue(self, issue: str) -> bool:
        """Attempt automated resolution for a given issue."""
        action = ""
        success = False
        if issue == "high_response_time":
            bots = {row[0] for row in self.metrics.fetch(5).itertuples(index=False)}
            DynamicResourceAllocator(self.metrics).allocate(bots)
            action = "reallocate_resources"
            success = True
        elif issue == "error_rate":
            self.error_bot.monitor()
            action = "error_bot_monitor"
            success = True
        elif issue == "discrepancies_detected":
            self._reroute_tasks()
            action = "reroute"
            success = True
        self.log.add(ResolutionRecord(issue, action, success))
        if not success:
            self.error_bot.handle_error(issue)
        return success

    def run(self) -> None:
        """Diagnose and resolve issues."""
        issues = self.diagnose()
        for i in issues:
            self.resolve_issue(i)


__all__ = ["ResolutionRecord", "ResolutionDB", "DiagnosticManager"]
