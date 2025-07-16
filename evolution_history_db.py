from __future__ import annotations

"""Store history of evolution cycles and their outcomes."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class EvolutionEvent:
    """Record of a single evolution cycle."""

    action: str
    before_metric: float
    after_metric: float
    roi: float
    predicted_roi: float = 0.0
    efficiency: float = 0.0
    bottleneck: float = 0.0
    patch_id: int | None = None
    workflow_id: int | None = None
    ts: str = datetime.utcnow().isoformat()
    trending_topic: str | None = None


class EvolutionHistoryDB:
    """SQLite-backed store for evolution history."""

    def __init__(self, path: Path | str = "evolution_history.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_history(
                action TEXT,
                before_metric REAL,
                after_metric REAL,
                roi REAL,
                predicted_roi REAL DEFAULT 0,
                efficiency REAL DEFAULT 0,
                bottleneck REAL DEFAULT 0,
                patch_id INTEGER,
                workflow_id INTEGER,
                ts TEXT,
                trending_topic TEXT
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(evolution_history)").fetchall()]
        if "efficiency" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN efficiency REAL DEFAULT 0")
        if "bottleneck" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN bottleneck REAL DEFAULT 0")
        if "predicted_roi" not in cols:
            self.conn.execute(
                "ALTER TABLE evolution_history ADD COLUMN predicted_roi REAL DEFAULT 0"
            )
        if "patch_id" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN patch_id INTEGER")
        if "workflow_id" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN workflow_id INTEGER")
        if "trending_topic" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN trending_topic TEXT")
        self.conn.commit()

    def add(self, event: EvolutionEvent) -> int:
        cur = self.conn.execute(
            "INSERT INTO evolution_history(action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                event.action,
                event.before_metric,
                event.after_metric,
                event.roi,
                event.predicted_roi,
                event.efficiency,
                event.bottleneck,
                event.patch_id,
                event.workflow_id,
                event.ts,
                event.trending_topic,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_cycle(self, rowid: int, efficiency: float, bottleneck: float) -> None:
        self.conn.execute(
            "UPDATE evolution_history SET efficiency=?, bottleneck=? WHERE rowid=?",
            (efficiency, bottleneck, rowid),
        )
        self.conn.commit()

    def fetch(
        self, limit: int = 50
    ) -> List[Tuple[str, float, float, float, float, float, float, int | None, int | None, str, str | None]]:
        cur = self.conn.execute(
            "SELECT action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic FROM evolution_history ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def summary(self, limit: int = 50) -> dict[str, float]:
        """Return simple stats for recent evolution events."""
        rows = self.fetch(limit)
        if not rows:
            return {"count": 0, "avg_roi": 0.0, "avg_delta": 0.0}
        count = len(rows)
        avg_roi = sum(r[3] for r in rows) / count
        avg_delta = sum(r[2] - r[1] for r in rows) / count
        return {"count": count, "avg_roi": avg_roi, "avg_delta": avg_delta}

    def averages(self, limit: int = 5) -> dict[str, float]:
        """Return average ROI delta and efficiency for recent cycles."""
        rows = self.fetch(limit)
        if not rows:
            return {"avg_roi_delta": 0.0, "avg_efficiency": 0.0}
        count = len(rows)
        avg_roi_delta = sum(r[2] - r[1] for r in rows) / count
        avg_eff = sum(r[5] for r in rows) / count
        return {"avg_roi_delta": avg_roi_delta, "avg_efficiency": avg_eff}


__all__ = ["EvolutionEvent", "EvolutionHistoryDB"]

