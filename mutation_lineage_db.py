from __future__ import annotations

"""Store mutation lineage events for traversal."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import sqlite3


@dataclass
class MutationEvent:
    """Record of a code mutation and its lineage context."""

    id: int | None = None
    parent_id: int | None = None
    workflow: str = ""
    change: str = ""
    reason: str = ""
    trigger: str = ""
    before_metric: float = 0.0
    after_metric: float = 0.0
    roi: float = 0.0
    timestamp: str = datetime.utcnow().isoformat()


class MutationLineageDB:
    """SQLite-backed helper for :class:`MutationEvent` records."""

    def __init__(self, path: Path | str = "mutation_lineage.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mutation_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER,
                workflow TEXT,
                change TEXT,
                reason TEXT,
                trigger TEXT,
                before_metric REAL,
                after_metric REAL,
                roi REAL,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def add(self, event: MutationEvent) -> int:
        """Insert *event* and return the new row id."""
        cur = self.conn.execute(
            "INSERT INTO mutation_events(parent_id, workflow, change, reason, trigger, before_metric, after_metric, roi, ts) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                event.parent_id,
                event.workflow,
                event.change,
                event.reason,
                event.trigger,
                event.before_metric,
                event.after_metric,
                event.roi,
                event.timestamp,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    def children(self, parent_id: int) -> List[MutationEvent]:
        """Return direct children of ``parent_id``."""
        cur = self.conn.execute(
            "SELECT id, parent_id, workflow, change, reason, trigger, before_metric, after_metric, roi, ts FROM mutation_events WHERE parent_id=?",
            (parent_id,),
        )
        rows = cur.fetchall()
        return [MutationEvent(*row) for row in rows]

    # ------------------------------------------------------------------
    def ancestors(self, event_id: int) -> List[MutationEvent]:
        """Return lineage from root to ``event_id``."""
        lineage: List[MutationEvent] = []
        current: Optional[int] = event_id
        while current is not None:
            row = self.conn.execute(
                "SELECT id, parent_id, workflow, change, reason, trigger, before_metric, after_metric, roi, ts FROM mutation_events WHERE id=?",
                (current,),
            ).fetchone()
            if not row:
                break
            event = MutationEvent(*row)
            lineage.append(event)
            current = event.parent_id
        return list(reversed(lineage))


__all__ = ["MutationEvent", "MutationLineageDB"]
