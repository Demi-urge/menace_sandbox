"""Persistent storage for workflow ROI evaluation results.

This module provides a thin wrapper around :class:`sqlite3.Connection`
managed by :class:`~menace_sandbox.db_router.DBRouter`.  Each recorded
evaluation captures aggregate workflow metrics along with per-module deltas
encoded as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json

from db_router import DBRouter, LOCAL_TABLES

# ``workflow_results`` is always treated as a local table so unit tests can
# operate on ephemeral SQLite files without touching shared resources.
LOCAL_TABLES.add("workflow_results")


@dataclass
class ROIResult:
    """Aggregate ROI metrics for a workflow evaluation."""

    workflow_id: str
    run_id: str
    timestamp: str
    runtime: float
    success_rate: float
    roi_gain: float
    workflow_synergy_score: float
    bottleneck_index: float
    patchability_score: float
    module_deltas: Dict[str, Dict[str, float]]


class ROIResultsDB:
    """Lightweight SQLite helper for workflow ROI results.

    The schema mirrors the specification from the user instructions. Each call
    to :meth:`log_result` inserts a single row capturing aggregate metrics and
    JSON encoded per-module deltas.
    """

    def __init__(self, path: str | Path = "roi_results.db", *, router: DBRouter | None = None) -> None:
        self.path = Path(path)
        self.router = router or DBRouter("workflow_results", str(self.path), str(self.path))
        self.conn = self.router.get_connection("workflow_results", operation="write")
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_results(
                workflow_id TEXT,
                run_id TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                runtime REAL,
                success_rate REAL,
                roi_gain REAL,
                workflow_synergy_score REAL,
                bottleneck_index REAL,
                patchability_score REAL,
                module_deltas TEXT
            )
            """,
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflow_results_wf_run
                ON workflow_results(workflow_id, run_id)
            """,
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS module_attribution(
                module TEXT PRIMARY KEY,
                roi_delta REAL,
                bottleneck REAL,
                runs INTEGER
            )
            """,
        )
        # migrate legacy ``roi_results`` table if present
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roi_results'"
        )
        if cur.fetchone():
            cur.execute(
                """
                INSERT INTO workflow_results(
                    workflow_id, run_id, timestamp, runtime, success_rate,
                    roi_gain, workflow_synergy_score, bottleneck_index,
                    patchability_score, module_deltas
                )
                SELECT
                    workflow_id, run_id, ts, runtime, success_rate,
                    roi_gain, workflow_synergy_score, bottleneck_index,
                    patchability_score, module_deltas
                FROM roi_results
                """,
            )
            cur.execute("DROP TABLE roi_results")
        self.conn.commit()

    # ------------------------------------------------------------------
    def log_result(
        self,
        *,
        workflow_id: str,
        run_id: str,
        runtime: float,
        success_rate: float,
        roi_gain: float,
        workflow_synergy_score: float,
        bottleneck_index: float,
        patchability_score: float,
        module_deltas: Dict[str, Dict[str, float]] | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Insert a workflow evaluation result."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO workflow_results(
                workflow_id, run_id, timestamp, runtime, success_rate,
                roi_gain, workflow_synergy_score, bottleneck_index,
                patchability_score, module_deltas
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                workflow_id,
                run_id,
                timestamp or datetime.utcnow().isoformat(),
                runtime,
                success_rate,
                roi_gain,
                workflow_synergy_score,
                bottleneck_index,
                patchability_score,
                json.dumps(module_deltas or {}, sort_keys=True),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    # ------------------------------------------------------------------
    def log_module_attribution(self, module: str, roi_delta: float, bottleneck: float) -> None:
        """Update per-module attribution stats."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO module_attribution(module, roi_delta, bottleneck, runs)
            VALUES(?,?,?,1)
            ON CONFLICT(module) DO UPDATE SET
                roi_delta=roi_delta + excluded.roi_delta,
                bottleneck=bottleneck + excluded.bottleneck,
                runs=runs + 1
            """,
            (module, roi_delta, bottleneck),
        )
        self.conn.commit()

    def fetch_module_attribution(self) -> Dict[str, Dict[str, float]]:
        """Return cumulative per-module attribution metrics."""

        cur = self.conn.cursor()
        cur.execute("SELECT module, roi_delta, bottleneck, runs FROM module_attribution")
        rows = cur.fetchall()
        return {
            str(m): {"roi_delta": float(r), "bottleneck": float(b), "runs": int(n)}
            for m, r, b, n in rows
        }

    # backward compatibility -------------------------------------------------
    def add_result(self, **kwargs: Any) -> int:  # pragma: no cover - legacy alias
        return self.log_result(**kwargs)

    def add(self, result: ROIResult) -> int:  # pragma: no cover - legacy alias
        return self.log_result(
            workflow_id=result.workflow_id,
            run_id=result.run_id,
            runtime=result.runtime,
            success_rate=result.success_rate,
            roi_gain=result.roi_gain,
            workflow_synergy_score=result.workflow_synergy_score,
            bottleneck_index=result.bottleneck_index,
            patchability_score=result.patchability_score,
            module_deltas=result.module_deltas,
            timestamp=result.timestamp,
        )

    # ------------------------------------------------------------------
    def fetch_results(self, workflow_id: str, run_id: str | None = None) -> List[ROIResult]:
        """Return logged results for ``workflow_id``.

        If ``run_id`` is provided, only matching entries are returned.
        """

        cur = self.conn.cursor()
        if run_id is None:
            cur.execute(
                "SELECT workflow_id, run_id, timestamp, runtime, success_rate, roi_gain, workflow_synergy_score, bottleneck_index, patchability_score, module_deltas FROM workflow_results WHERE workflow_id=? ORDER BY timestamp",
                (workflow_id,),
            )
        else:
            cur.execute(
                "SELECT workflow_id, run_id, timestamp, runtime, success_rate, roi_gain, workflow_synergy_score, bottleneck_index, patchability_score, module_deltas FROM workflow_results WHERE workflow_id=? AND run_id=? ORDER BY timestamp",
                (workflow_id, run_id),
            )
        rows = cur.fetchall()
        results: List[ROIResult] = []
        for row in rows:
            results.append(
                ROIResult(
                    workflow_id=row[0],
                    run_id=row[1],
                    timestamp=str(row[2]),
                    runtime=float(row[3]),
                    success_rate=float(row[4]),
                    roi_gain=float(row[5]),
                    workflow_synergy_score=float(row[6]),
                    bottleneck_index=float(row[7]),
                    patchability_score=float(row[8]),
                    module_deltas=json.loads(row[9] or "{}"),
                )
            )
        return results

    # ------------------------------------------------------------------
    def module_impact_report(self, workflow_id: str, run_id: str) -> Dict[str, Dict[str, float]]:
        """Return modules grouped by improvement sign for ``run_id``."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT run_id, module_deltas FROM workflow_results WHERE workflow_id=? ORDER BY timestamp",
            (workflow_id,),
        )
        rows = cur.fetchall()

        prev: Dict[str, float] = {}
        for r_id, deltas_json in rows:
            deltas = json.loads(deltas_json or "{}")
            if r_id == run_id:
                improved: Dict[str, float] = {}
                regressed: Dict[str, float] = {}
                for mod, metrics in deltas.items():
                    curr = float(metrics.get("roi_delta", 0.0))
                    diff = curr - prev.get(mod, 0.0)
                    if diff >= 0:
                        improved[mod] = diff
                    else:
                        regressed[mod] = diff
                return {"improved": improved, "regressed": regressed}
            for mod, metrics in deltas.items():
                prev[mod] = float(metrics.get("roi_delta", 0.0))

        return {"improved": {}, "regressed": {}}


def module_impact_report(workflow_id: str, run_id: str, db_path: str | Path = "roi_results.db") -> Dict[str, Dict[str, float]]:
    """Convenience wrapper returning module impact report from ``db_path``."""

    db = ROIResultsDB(db_path)
    return db.module_impact_report(workflow_id, run_id)


__all__ = ["ROIResult", "ROIResultsDB", "module_impact_report"]

