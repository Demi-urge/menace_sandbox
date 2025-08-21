"""Simple telemetry backend for ROI predictions and drift metrics."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional


class TelemetryBackend:
    """Persist ROI prediction telemetry to SQLite."""

    def __init__(self, db_path: str = "telemetry.db") -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS roi_telemetry (
                        workflow_id TEXT,
                        predicted REAL,
                        actual REAL,
                        confidence REAL,
                        scenario TEXT,
                        scenario_deltas TEXT,
                        drift_flag INTEGER,
                        readiness REAL,
                        ts DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                for stmt in (
                    "ALTER TABLE roi_telemetry ADD COLUMN readiness REAL",
                    "ALTER TABLE roi_telemetry ADD COLUMN scenario TEXT",
                ):
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError:
                        pass
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def log_prediction(
        self,
        workflow_id: str | None,
        predicted: float | None,
        actual: float | None,
        confidence: float | None,
        scenario_deltas: Optional[Dict[str, float]],
        drift_flag: bool,
        readiness: float | None,
        scenario: str | None = None,
        ts: str | None = None,
    ) -> None:
        """Store a prediction outcome and associated metrics."""

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                params = (
                    workflow_id,
                    None if predicted is None else float(predicted),
                    None if actual is None else float(actual),
                    None if confidence is None else float(confidence),
                    scenario,
                    json.dumps(scenario_deltas or {}),
                    int(bool(drift_flag)),
                    None if readiness is None else float(readiness),
                )
                if ts is None:
                    conn.execute(
                        "INSERT INTO roi_telemetry (workflow_id, predicted, actual, confidence, scenario, scenario_deltas, drift_flag, readiness) VALUES (?,?,?,?,?,?,?,?)",
                        params,
                    )
                else:
                    conn.execute(
                        "INSERT INTO roi_telemetry (workflow_id, predicted, actual, confidence, scenario, scenario_deltas, drift_flag, readiness, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                        params + (ts,),
                    )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def fetch_history(
        self,
        workflow_id: str | None = None,
        *,
        scenario: str | None = None,
        start_ts: str | None = None,
        end_ts: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return logged prediction history filtered by workflow or scenario."""

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            base = (
                "SELECT workflow_id, predicted, actual, confidence, scenario, "
                "scenario_deltas, drift_flag, readiness, ts FROM roi_telemetry"
            )
            clauses: List[str] = []
            params: List[Any] = []
            if workflow_id is not None:
                clauses.append("workflow_id = ?")
                params.append(workflow_id)
            if scenario is not None:
                clauses.append("scenario = ?")
                params.append(scenario)
            if start_ts is not None:
                clauses.append("ts >= ?")
                params.append(start_ts)
            if end_ts is not None:
                clauses.append("ts <= ?")
                params.append(end_ts)
            if clauses:
                base += " WHERE " + " AND ".join(clauses)
            base += " ORDER BY ts"
            cur.execute(base, params)
            rows = cur.fetchall()
            result: List[Dict[str, Any]] = []
            for wf, pred, act, conf, scen, deltas, drift, ready, ts in rows:
                result.append(
                    {
                        "workflow_id": wf,
                        "predicted": pred,
                        "actual": act,
                        "confidence": conf,
                        "scenario": scen,
                        "scenario_deltas": json.loads(deltas) if deltas else {},
                        "drift_flag": bool(drift),
                        "readiness": ready,
                        "ts": ts,
                    }
                )
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def fetch_drift_metrics(self, workflow_id: str | None = None) -> Dict[str, Any]:
        """Return summary drift statistics for ``workflow_id``."""

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            if workflow_id is None:
                cur.execute("SELECT drift_flag FROM roi_telemetry")
            else:
                cur.execute(
                    "SELECT drift_flag FROM roi_telemetry WHERE workflow_id = ?",
                    (workflow_id,),
                )
            flags = [bool(x[0]) for x in cur.fetchall()]
            total = len(flags)
            drifted = sum(flags)
            rate = drifted / total if total else 0.0
            return {"total": total, "drifted": drifted, "rate": rate}
        finally:
            conn.close()
