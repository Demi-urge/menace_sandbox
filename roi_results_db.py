"""Persistent storage for workflow ROI evaluation results.

This module provides a thin wrapper around :class:`sqlite3.Connection`
managed by :class:`~menace_sandbox.db_router.DBRouter`.  Each recorded
evaluation captures aggregate workflow metrics along with per-module deltas
encoded as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json

from db_router import DBRouter, LOCAL_TABLES

# ``roi_results`` is always treated as a local table so unit tests can operate
# on ephemeral SQLite files without touching shared resources.
LOCAL_TABLES.add("roi_results")


@dataclass
class ROIResult:
    """Aggregate ROI metrics for a workflow evaluation."""

    workflow_id: str
    run_id: str
    runtime: float
    success_rate: float
    roi_gain: float
    workflow_synergy_score: float
    bottleneck_index: float
    patchability_score: float
    module_deltas: Dict[str, Dict[str, float]]


class ROIResultsDB:
    """Lightweight SQLite helper for workflow ROI results.

    The schema mirrors the specification from the user instructions.  Each
    call to :meth:`add_result` inserts a single row capturing aggregate metrics
    and JSON encoded per-module deltas.
    """

    def __init__(self, path: str | Path = "roi_results.db", *, router: DBRouter | None = None) -> None:
        self.path = Path(path)
        self.router = router or DBRouter("roi_results", str(self.path), str(self.path))
        self.conn = self.router.get_connection("roi_results", operation="write")
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS roi_results(
                workflow_id TEXT,
                run_id TEXT,
                runtime REAL,
                success_rate REAL,
                roi_gain REAL,
                workflow_synergy_score REAL,
                bottleneck_index REAL,
                patchability_score REAL,
                module_deltas TEXT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_roi_results_wf_run ON roi_results(workflow_id, run_id)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def add_result(
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
    ) -> int:
        """Insert a workflow evaluation result.

        Parameters
        ----------
        workflow_id:
            Identifier of the evaluated workflow.
        run_id:
            Unique identifier for this evaluation run.
        runtime, success_rate, roi_gain, workflow_synergy_score,
        bottleneck_index, patchability_score:
            Aggregate metrics from the evaluation.
        module_deltas:
            Mapping of module name to metric dictionary.  The mapping is JSON
            encoded before storage.
        """

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO roi_results(
                workflow_id, run_id, runtime, success_rate, roi_gain,
                workflow_synergy_score, bottleneck_index, patchability_score,
                module_deltas
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                workflow_id,
                run_id,
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
    def module_impact_report(self, workflow_id: str, run_id: str) -> Dict[str, Dict[str, float]]:
        """Return modules grouped by improvement sign for ``run_id``."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT run_id, module_deltas FROM roi_results WHERE workflow_id=? ORDER BY ts",
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

