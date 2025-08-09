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
    reason: str = ""
    trigger: str = ""
    performance: float = 0.0
    parent_event_id: int | None = None


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
                trending_topic TEXT,
                reason TEXT,
                "trigger" TEXT,
                performance REAL DEFAULT 0,
                parent_event_id INTEGER
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
        if "reason" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN reason TEXT")
        if "trigger" not in cols:
            self.conn.execute('ALTER TABLE evolution_history ADD COLUMN "trigger" TEXT')
        if "performance" not in cols:
            self.conn.execute("ALTER TABLE evolution_history ADD COLUMN performance REAL DEFAULT 0")
        if "parent_event_id" not in cols:
            if "parent_id" in cols:
                self.conn.execute(
                    "ALTER TABLE evolution_history RENAME COLUMN parent_id TO parent_event_id"
                )
            else:
                self.conn.execute(
                    "ALTER TABLE evolution_history ADD COLUMN parent_event_id INTEGER"
                )
        self.conn.commit()

    def add(self, event: EvolutionEvent) -> int:
        cur = self.conn.execute(
            'INSERT INTO evolution_history(action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, reason, "trigger", performance, parent_event_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
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
                event.reason,
                event.trigger,
                event.performance,
                event.parent_event_id,
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
    ) -> List[
        Tuple[
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            int | None,
            int | None,
            str,
            str | None,
            str,
            str,
            float,
            int | None,
        ]
    ]:
        cur = self.conn.execute(
            'SELECT action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, reason, "trigger", performance, parent_event_id FROM evolution_history ORDER BY ts DESC LIMIT ?',
            (limit,),
        )
        return cur.fetchall()

    def fetch_children(
        self, parent_event_id: int
    ) -> List[
        Tuple[
            int,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            int | None,
            int | None,
            str,
            str | None,
            str,
            str,
            float,
            int | None,
        ]
    ]:
        cur = self.conn.execute(
            'SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, reason, "trigger", performance, parent_event_id FROM evolution_history WHERE parent_event_id=?',
            (parent_event_id,),
        )
        return cur.fetchall()

    # ------------------------------------------------------------------
    def subtree(self, event_id: int) -> dict | None:
        """Return the mutation subtree rooted at ``event_id``."""

        fields = [
            "rowid",
            "action",
            "before_metric",
            "after_metric",
            "roi",
            "predicted_roi",
            "efficiency",
            "bottleneck",
            "patch_id",
            "workflow_id",
            "ts",
            "trending_topic",
            "reason",
            "trigger",
            "performance",
            "parent_event_id",
        ]

        def build(row: Tuple) -> dict:
            node = dict(zip(fields, row))
            kids = self.fetch_children(row[0])
            node["children"] = [build(child) for child in kids]
            return node

        row = self.conn.execute(
            'SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, reason, "trigger", performance, parent_event_id FROM evolution_history WHERE rowid=?',
            (event_id,),
        ).fetchone()
        if not row:
            return None
        return build(row)

    # ------------------------------------------------------------------
    def spawn_variant(
        self,
        parent_event_id: int,
        action: str,
        *,
        reason: str = "spawn_variant",
        trigger: str = "spawn_variant",
    ) -> int:
        """Create a new evolution event branching from ``parent_event_id``."""

        row = self.conn.execute(
            "SELECT workflow_id FROM evolution_history WHERE rowid=?",
            (parent_event_id,),
        ).fetchone()
        workflow_id = row[0] if row else None
        event = EvolutionEvent(
            action=action,
            before_metric=0.0,
            after_metric=0.0,
            roi=0.0,
            workflow_id=workflow_id,
            reason=reason,
            trigger=trigger,
            parent_event_id=parent_event_id,
        )
        return self.add(event)

    def lineage_tree(self, workflow_id: int) -> List[dict]:
        cur = self.conn.execute(
            'SELECT rowid, action, before_metric, after_metric, roi, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, ts, trending_topic, reason, "trigger", performance, parent_event_id FROM evolution_history WHERE workflow_id=?',
            (workflow_id,),
        )
        rows = cur.fetchall()
        by_parent: dict[int | None, list] = {}
        for row in rows:
            by_parent.setdefault(row[-1], []).append(row)

        def build(pid: int | None) -> List[dict]:
            children: List[dict] = []
            for r in by_parent.get(pid, []):
                children.append(
                    {
                        "rowid": r[0],
                        "action": r[1],
                        "before_metric": r[2],
                        "after_metric": r[3],
                        "roi": r[4],
                        "predicted_roi": r[5],
                        "efficiency": r[6],
                        "bottleneck": r[7],
                        "patch_id": r[8],
                        "workflow_id": r[9],
                        "ts": r[10],
                        "trending_topic": r[11],
                        "reason": r[12],
                        "trigger": r[13],
                        "performance": r[14],
                        "parent_event_id": r[15],
                        "children": build(r[0]),
                    }
                )
            return children

        return build(None)

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

