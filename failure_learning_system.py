"""Failure Learning System storing model failures."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from . import bot_planning_bot as bpb
from . import capital_management_bot as cmb


@dataclass
class FailureRecord:
    """Detailed failure event data."""

    model_id: str
    cause: str
    features: str
    demographics: str
    profitability: float
    retention: float
    cac: float
    roi: float
    ts: str = datetime.utcnow().isoformat()


class DiscrepancyDB:
    """SQLite store for failures and discrepancy detections."""

    def __init__(self, path: Path | str = "failures.db") -> None:
        # allow the connection to be shared across threads because the
        # failure learning system may be accessed by background workers
        # running in different threads
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS failures(
                model_id TEXT,
                cause TEXT,
                features TEXT,
                demographics TEXT,
                profitability REAL,
                retention REAL,
                cac REAL,
                roi REAL,
                ts TEXT
            )
            """,
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections(
                rule TEXT,
                message TEXT,
                severity REAL,
                workflow TEXT,
                ts TEXT
            )
            """,
        )
        self.conn.commit()

    def log(self, rec: FailureRecord) -> None:
        self.conn.execute(
            "INSERT INTO failures VALUES(?,?,?,?,?,?,?,?,?)",
            (
                rec.model_id,
                rec.cause,
                rec.features,
                rec.demographics,
                rec.profitability,
                rec.retention,
                rec.cac,
                rec.roi,
                rec.ts,
            ),
        )
        self.conn.commit()

    def fetch(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM failures", self.conn)

    def log_detection(
        self, rule: str, severity: float, message: str, workflow: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO detections(rule, message, severity, workflow, ts) VALUES(?,?,?,?,?)",
            (
                rule,
                message,
                severity,
                workflow or "",
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def fetch_detections(self, min_severity: float = 0.0) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT rule, message, severity, workflow, ts FROM detections WHERE severity >= ?",
            self.conn,
            params=(min_severity,),
        )


class FailureLearningSystem:
    """Record failures and derive planning and funding insights."""

    def __init__(self, db: DiscrepancyDB | None = None) -> None:
        self.db = db or DiscrepancyDB()

    def record_failure(self, rec: FailureRecord) -> None:
        self.db.log(rec)

    def risk_features(self) -> Dict[str, int]:
        df = self.db.fetch()
        counts: Dict[str, int] = {}
        if "features" not in df:
            return counts
        for feats in df["features"]:
            for f in feats.split(","):
                f = f.strip()
                if not f:
                    continue
                counts[f] = counts.get(f, 0) + 1
        return counts

    def failure_score(self, model_id: str) -> float:
        df = self.db.fetch()
        total = len(df)
        if total == 0:
            return 0.0
        return float(len(df[df["model_id"] == model_id])) / float(total)

    def advise_planner(self, planner: bpb.BotPlanningBot) -> List[str]:
        counts = self.risk_features()
        return [feat for feat, c in counts.items() if c > 1]

    def advise_capital(self, manager: cmb.CapitalManagementBot, model_id: str) -> float:
        return self.failure_score(model_id)


__all__ = ["FailureRecord", "DiscrepancyDB", "FailureLearningSystem"]
