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
                        scenario_deltas TEXT,
                        drift_flag INTEGER,
                        readiness REAL,
                        ts DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                try:
                    conn.execute(
                        "ALTER TABLE roi_telemetry ADD COLUMN readiness REAL"
                    )
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
    ) -> None:
        """Store a prediction outcome and associated metrics."""

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO roi_telemetry (workflow_id, predicted, actual, confidence, scenario_deltas, drift_flag, readiness) VALUES (?,?,?,?,?,?,?)",
                    (
                        workflow_id,
                        None if predicted is None else float(predicted),
                        None if actual is None else float(actual),
                        None if confidence is None else float(confidence),
                        json.dumps(scenario_deltas or {}),
                        int(bool(drift_flag)),
                        None if readiness is None else float(readiness),
                    ),
                )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def fetch_history(self, workflow_id: str | None = None) -> List[Dict[str, Any]]:
        """Return logged prediction history for ``workflow_id``."""

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            if workflow_id is None:
                cur.execute(
                    "SELECT workflow_id, predicted, actual, confidence, scenario_deltas, drift_flag, readiness, ts FROM roi_telemetry ORDER BY ts"
                )
            else:
                cur.execute(
                    "SELECT workflow_id, predicted, actual, confidence, scenario_deltas, drift_flag, readiness, ts FROM roi_telemetry WHERE workflow_id = ? ORDER BY ts",
                    (workflow_id,),
                )
            rows = cur.fetchall()
            result: List[Dict[str, Any]] = []
            for wf, pred, act, conf, scen, drift, ready, ts in rows:
                result.append(
                    {
                        "workflow_id": wf,
                        "predicted": pred,
                        "actual": act,
                        "confidence": conf,
                        "scenario_deltas": json.loads(scen) if scen else {},
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
