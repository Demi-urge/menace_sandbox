from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sqlite3

from db_router import DBRouter, LOCAL_TABLES

# Ensure router treats ``roi_results`` as a local table so test environments can
# operate on ephemeral SQLite files without external dependencies.
LOCAL_TABLES.add("roi_results")


@dataclass
class ROIResult:
    """Record of a single module or workflow ROI measurement."""

    workflow_id: str
    run_id: str
    module: Optional[str]
    runtime: float
    success_rate: float
    roi_gain: float
    workflow_synergy_score: Optional[float] = None
    bottleneck_index: Optional[float] = None
    patchability_score: Optional[float] = None
    ts: str = datetime.utcnow().isoformat()


class ROIResultsDB:
    """SQLite helper providing longitudinal ROI result storage and queries."""

    def __init__(
        self,
        db_path: str | Path = "roi_results.db",
        router: DBRouter | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        # Reuse provided router to share connections with other helpers, falling
        # back to a standalone ``DBRouter`` if necessary.
        self.router = router or DBRouter(
            "roi_results", str(self.db_path), str(self.db_path)
        )
        self.conn = self.router.get_connection("roi_results", operation="write")
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS roi_results(
                id INTEGER PRIMARY KEY,
                workflow_id TEXT,
                run_id TEXT,
                module TEXT,
                runtime REAL,
                success_rate REAL,
                roi_gain REAL,
                workflow_synergy_score REAL,
                bottleneck_index REAL,
                patchability_score REAL,
                ts TEXT
            )
            """,
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_roi_results_wf_run_mod
                ON roi_results(workflow_id, run_id, module)
            """,
        )
        self.conn.commit()

    # write ---------------------------------------------------------------
    def add(self, rec: ROIResult) -> None:
        """Persist an :class:`ROIResult` record."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO roi_results(
                workflow_id, run_id, module, runtime, success_rate, roi_gain,
                workflow_synergy_score, bottleneck_index, patchability_score, ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.workflow_id,
                rec.run_id,
                rec.module,
                rec.runtime,
                rec.success_rate,
                rec.roi_gain,
                rec.workflow_synergy_score,
                rec.bottleneck_index,
                rec.patchability_score,
                rec.ts,
            ),
        )
        self.conn.commit()

    # read ----------------------------------------------------------------
    def fetch_runs(
        self, workflow_id: str, module: Optional[str] = None
    ) -> List[ROIResult]:
        """Return all recorded runs for ``workflow_id`` optionally filtered by module."""

        cur = self.conn.cursor()
        base = (
            "SELECT workflow_id, run_id, module, runtime, success_rate, roi_gain, "
            "workflow_synergy_score, bottleneck_index, patchability_score, ts "
            "FROM roi_results WHERE workflow_id=?"
        )
        params: list[object] = [workflow_id]
        if module is not None:
            base += " AND module=?"
            params.append(module)
        base += " ORDER BY ts"
        cur.execute(base, params)
        rows = cur.fetchall()
        return [ROIResult(*row) for row in rows]

    def trend(self, workflow_id: str, metric: str, module: Optional[str] = None) -> float:
        """Return linear trend slope for ``metric`` across runs.

        The computation uses ordinary least squares on the sequential runs.  If
        fewer than two points are present, ``0.0`` is returned.
        """

        rows = self.fetch_runs(workflow_id, module)
        values = [getattr(r, metric) for r in rows if getattr(r, metric) is not None]
        n = len(values)
        if n < 2:
            return 0.0
        mean_x = (n - 1) / 2
        mean_y = sum(values) / n
        num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
        den = sum((i - mean_x) ** 2 for i in range(n))
        return num / den if den else 0.0

    def module_impact_report(self, workflow_id: str, run_id: str) -> Dict[str, Dict[str, float]]:
        """Return modules grouped by improvement sign for ``run_id``."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT module, roi_gain FROM roi_results WHERE workflow_id=? AND run_id=? AND module IS NOT NULL",
            (workflow_id, run_id),
        )
        current = {str(m): float(g) for m, g in cur.fetchall()}
        improved: Dict[str, float] = {}
        regressed: Dict[str, float] = {}
        for mod in current:
            history = self.fetch_runs(workflow_id, mod)
            cumulative = 0.0
            prev_total = 0.0
            cur_total = 0.0
            for rec in history:
                cumulative += rec.roi_gain
                if rec.run_id == run_id:
                    cur_total = cumulative
                    break
                prev_total = cumulative
            delta = cur_total - prev_total
            if delta >= 0:
                improved[mod] = delta
            else:
                regressed[mod] = delta
        return {"improved": improved, "regressed": regressed}


def module_impact_report(
    workflow_id: str, run_id: str, db_path: str | Path = "roi_results.db"
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper returning module impact report from ``db_path``."""

    db = ROIResultsDB(db_path)
    return db.module_impact_report(workflow_id, run_id)


__all__ = ["ROIResult", "ROIResultsDB", "module_impact_report"]
