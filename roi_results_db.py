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
from statistics import fmean, pvariance

from dynamic_path_router import resolve_path
from db_router import DBRouter, LOCAL_TABLES

# ``workflow_results`` is always treated as a local table so unit tests can
# operate on ephemeral SQLite files without touching shared resources.
LOCAL_TABLES.add("workflow_results")
LOCAL_TABLES.add("workflow_module_deltas")


try:  # pragma: no cover - compute default ROI results DB path
    _ROI_RESULTS_DB_PATH = resolve_path("roi_results.db")
except FileNotFoundError:  # pragma: no cover - file may not exist yet
    _ROI_RESULTS_DB_PATH = resolve_path(".") / "roi_results.db"


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
    code_entropy: float = 0.0
    entropy_delta: float = 0.0
    failure_reason: str | None = None


class ROIResultsDB:
    """Lightweight SQLite helper for workflow ROI results.

    The schema mirrors the specification from the user instructions. Each call
    to :meth:`log_result` inserts a single row capturing aggregate metrics and
    JSON encoded per-module deltas.
    """

    def __init__(
        self,
        path: str | Path = _ROI_RESULTS_DB_PATH,
        *,
        router: DBRouter | None = None,
        window: int = 5,
    ) -> None:
        try:
            self.path = Path(resolve_path(str(path)))
        except FileNotFoundError:
            self.path = Path(path)
        self.router = router or DBRouter(
            "workflow_results", str(self.path), str(self.path)
        )
        self.conn = self.router.get_connection("workflow_results", operation="write")
        self.ma_window = window
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
                code_entropy REAL DEFAULT 0.0,
                entropy_delta REAL DEFAULT 0.0,
                module_deltas TEXT,
                failure_reason TEXT
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
            CREATE TABLE IF NOT EXISTS workflow_module_deltas(
                id INTEGER PRIMARY KEY,
                workflow_id TEXT,
                run_id TEXT,
                module TEXT,
                runtime REAL,
                success_rate REAL,
                roi_delta REAL,
                roi_delta_ma REAL DEFAULT 0.0,
                roi_delta_var REAL DEFAULT 0.0,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_module_deltas_wf_run_mod
                ON workflow_module_deltas(workflow_id, run_id, module)
            """,
        )
        existing = {
            r[1]
            for r in cur.execute("PRAGMA table_info(workflow_module_deltas)").fetchall()
        }
        if "success_rate" not in existing:
            cur.execute(
                "ALTER TABLE workflow_module_deltas ADD COLUMN success_rate REAL DEFAULT 0.0"
            )
        if "roi_delta_ma" not in existing:
            cur.execute(
                "ALTER TABLE workflow_module_deltas ADD COLUMN roi_delta_ma REAL DEFAULT 0.0"
            )
        if "roi_delta_var" not in existing:
            cur.execute(
                "ALTER TABLE workflow_module_deltas ADD COLUMN roi_delta_var REAL DEFAULT 0.0"
            )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_chain_stats(
                workflow_id TEXT PRIMARY KEY,
                moving_avg_roi REAL DEFAULT 0.0,
                last_roi REAL DEFAULT 0.0,
                non_positive_streak INTEGER DEFAULT 0
            )
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
                    patchability_score, module_deltas, failure_reason
                )
                SELECT
                    workflow_id, run_id, ts, runtime, success_rate,
                    roi_gain, workflow_synergy_score, bottleneck_index,
                    patchability_score, module_deltas, NULL
                FROM roi_results
                """,
            )
            cur.execute("DROP TABLE roi_results")
        existing = {
            r[1] for r in cur.execute("PRAGMA table_info(workflow_results)").fetchall()
        }
        if "code_entropy" not in existing:
            cur.execute(
                "ALTER TABLE workflow_results ADD COLUMN code_entropy REAL DEFAULT 0.0"
            )
        if "entropy_delta" not in existing:
            cur.execute(
                "ALTER TABLE workflow_results ADD COLUMN entropy_delta REAL DEFAULT 0.0"
            )
        if "failure_reason" not in existing:
            cur.execute("ALTER TABLE workflow_results ADD COLUMN failure_reason TEXT")
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
        code_entropy: float = 0.0,
        entropy_delta: float = 0.0,
        module_deltas: Dict[str, Dict[str, float]] | None = None,
        timestamp: str | None = None,
        failure_reason: str | None = None,
    ) -> int:
        """Insert a workflow evaluation result."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO workflow_results(
                    workflow_id, run_id, timestamp, runtime, success_rate,
                    roi_gain, workflow_synergy_score, bottleneck_index,
                    patchability_score, code_entropy, entropy_delta, module_deltas, failure_reason
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    code_entropy,
                    entropy_delta,
                    json.dumps(module_deltas or {}, sort_keys=True),
                    failure_reason,
                ),
            )
            if module_deltas:
                for mod, metrics in module_deltas.items():
                    self.log_module_delta(
                        workflow_id,
                        run_id,
                        mod,
                        float(metrics.get("runtime", 0.0)),
                        float(metrics.get("success_rate", 0.0)),
                        float(metrics.get("roi_delta", 0.0)),
                        commit=False,
                    )
            self._update_chain_stats(workflow_id)
            self.conn.commit()
            return int(cur.lastrowid or 0)
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------
    def _update_chain_stats(self, workflow_id: str, window: int | None = None) -> None:
        """Update moving average and streak for ``workflow_id``.

        The moving average is computed over the previous ``window`` results and
        excludes the most recent run. ``non_positive_streak`` counts consecutive
        runs where ``delta_roi`` is non-positive.
        """

        win = window or self.ma_window
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT roi_gain FROM workflow_results
            WHERE workflow_id=? ORDER BY timestamp DESC LIMIT ?
            """,
            (workflow_id, win + 1),
        )
        rows = [float(r[0]) for r in cur.fetchall()]
        if not rows:
            return
        current = rows[0]
        prev = rows[1:]
        moving_avg = fmean(prev) if prev else current
        delta = current - moving_avg
        cur.execute(
            "SELECT non_positive_streak FROM workflow_chain_stats WHERE workflow_id=?",
            (workflow_id,),
        )
        row = cur.fetchone()
        prev_streak = int(row[0]) if row else 0
        streak = prev_streak + 1 if delta <= 0 else 0
        cur.execute(
            """
            INSERT INTO workflow_chain_stats(
                workflow_id, moving_avg_roi, last_roi, non_positive_streak
            ) VALUES(?,?,?,?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                moving_avg_roi=excluded.moving_avg_roi,
                last_roi=excluded.last_roi,
                non_positive_streak=excluded.non_positive_streak
            """,
            (workflow_id, moving_avg, current, streak),
        )

    # ------------------------------------------------------------------
    def fetch_chain_stats(self, workflow_id: str) -> Dict[str, float]:
        """Return moving average, latest ROI and streak for ``workflow_id``."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT moving_avg_roi, last_roi, non_positive_streak "
            "FROM workflow_chain_stats WHERE workflow_id=?",
            (workflow_id,),
        )
        row = cur.fetchone()
        if row:
            ma, last, streak = row
            return {
                "moving_avg_roi": float(ma),
                "last_roi": float(last),
                "delta_roi": float(last - ma),
                "non_positive_streak": int(streak),
            }
        return {
            "moving_avg_roi": 0.0,
            "last_roi": 0.0,
            "delta_roi": 0.0,
            "non_positive_streak": 0,
        }

    # ------------------------------------------------------------------
    def fetch_stagnant_chains(self, min_streak: int = 3) -> Dict[str, int]:
        """Return workflow ids with ``non_positive_streak`` >= ``min_streak``."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT workflow_id, non_positive_streak "
            "FROM workflow_chain_stats WHERE non_positive_streak >= ?",
            (min_streak,),
        )
        return {str(w): int(s) for w, s in cur.fetchall()}

    # ------------------------------------------------------------------
    def projected_revenue(self) -> float:
        """Return total projected revenue recorded in the ROI results table."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roi_results'",
        )
        table = "roi_results" if cur.fetchone() else "workflow_results"
        cur.execute(f"SELECT SUM(roi_gain) FROM {table}")
        row = cur.fetchone()
        return float(row[0] or 0.0)

    # ------------------------------------------------------------------
    def log_module_delta(
        self,
        workflow_id: str,
        run_id: str,
        module: str,
        runtime: float,
        success_rate: float,
        roi_delta: float,
        *,
        window: int | None = None,
        commit: bool = True,
    ) -> None:
        """Persist per-module delta statistics for a workflow run.

        ``roi_delta_ma`` and ``roi_delta_var`` are derived from the latest
        ``window`` entries for the given workflow/module pair. If an entry
        already exists for ``workflow_id``/``run_id``/``module`` the call is
        ignored so callers may safely invoke this multiple times.
        """

        win = window or self.ma_window
        cur = self.conn.cursor()
        try:
            # skip if this run/module has already been recorded
            cur.execute(
                """
                SELECT 1 FROM workflow_module_deltas
                WHERE workflow_id=? AND run_id=? AND module=? LIMIT 1
                """,
                (workflow_id, run_id, module),
            )
            if cur.fetchone():
                return

            limit = max(win - 1, 0)
            cur.execute(
                """
                SELECT roi_delta FROM workflow_module_deltas
                WHERE workflow_id=? AND module=?
                ORDER BY ts DESC LIMIT ?
                """,
                (workflow_id, module, limit),
            )
            prev = [float(r[0]) for r in cur.fetchall()]
            values = prev + [roi_delta]
            roi_delta_ma = fmean(values)
            roi_delta_var = pvariance(values) if len(values) > 1 else 0.0
            cur.execute(
                """
                INSERT INTO workflow_module_deltas(
                    workflow_id, run_id, module, runtime, success_rate,
                    roi_delta, roi_delta_ma, roi_delta_var
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    workflow_id,
                    run_id,
                    module,
                    runtime,
                    success_rate,
                    roi_delta,
                    roi_delta_ma,
                    roi_delta_var,
                ),
            )
            if commit:
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # backward compatibility -------------------------------------------------
    def log_module_deltas(
        self,
        workflow_id: str,
        run_id: str,
        module: str,
        runtime: float,
        success_rate: float,
        roi_delta: float,
        *,
        window: int | None = None,
    ) -> None:  # pragma: no cover - legacy alias
        """Alias for :meth:`log_module_delta` for backwards compatibility."""

        self.log_module_delta(
            workflow_id,
            run_id,
            module,
            runtime,
            success_rate,
            roi_delta,
            window=window,
        )

    # ------------------------------------------------------------------
    def log_module_attribution(
        self, module: str, roi_delta: float, bottleneck: float
    ) -> None:
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
        cur.execute(
            "SELECT module, roi_delta, bottleneck, runs FROM module_attribution"
        )
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
            failure_reason=result.failure_reason,
        )

    # ------------------------------------------------------------------
    def fetch_results(
        self, workflow_id: str, run_id: str | None = None
    ) -> List[ROIResult]:
        """Return logged results for ``workflow_id``.

        If ``run_id`` is provided, only matching entries are returned.
        """

        cur = self.conn.cursor()
        if run_id is None:
            cur.execute(
                (
                    "SELECT workflow_id, run_id, timestamp, runtime, success_rate, roi_gain, "
                    "workflow_synergy_score, bottleneck_index, patchability_score, "
                    "module_deltas, failure_reason FROM workflow_results "
                    "WHERE workflow_id=? ORDER BY timestamp"
                ),
                (workflow_id,),
            )
        else:
            cur.execute(
                (
                    "SELECT workflow_id, run_id, timestamp, runtime, success_rate, roi_gain, "
                    "workflow_synergy_score, bottleneck_index, patchability_score, "
                    "module_deltas, failure_reason FROM workflow_results "
                    "WHERE workflow_id=? AND run_id=? ORDER BY timestamp"
                ),
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
                    failure_reason=row[10],
                )
            )
        return results

    # ------------------------------------------------------------------
    def fetch_trends(self, workflow_id: str) -> List[Dict[str, float]]:
        """Return time ordered aggregate metrics for ``workflow_id``."""

        cur = self.conn.cursor()
        cur.execute(
            (
                "SELECT timestamp, roi_gain, workflow_synergy_score, "
                "bottleneck_index, patchability_score "
                "FROM workflow_results WHERE workflow_id=? ORDER BY timestamp"
            ),
            (workflow_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "timestamp": str(ts),
                "roi_gain": float(roi),
                "workflow_synergy_score": float(syn),
                "bottleneck_index": float(bot),
                "patchability_score": float(patch),
            }
            for ts, roi, syn, bot, patch in rows
        ]

    # ------------------------------------------------------------------
    def fetch_module_trajectories(
        self, workflow_id: str, module: str | None = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """Return per-run trend metrics for modules in ``workflow_id``.

        Each trajectory entry includes the raw ROI delta along with its moving
        average and variance as stored in the ``roi_delta_ma`` and
        ``roi_delta_var`` columns of the ``workflow_module_deltas`` table.  The
        presence of these statistics enables analysis of patchability and
        volatility over time for individual modules.
        """

        cur = self.conn.cursor()
        if module is None:
            cur.execute(
                """
                SELECT module, run_id, success_rate, roi_delta, roi_delta_ma, roi_delta_var
                FROM workflow_module_deltas
                WHERE workflow_id=?
                ORDER BY ts, id
                """,
                (workflow_id,),
            )
        else:
            cur.execute(
                """
                SELECT module, run_id, success_rate, roi_delta, roi_delta_ma, roi_delta_var
                FROM workflow_module_deltas
                WHERE workflow_id=? AND module=?
                ORDER BY ts, id
                """,
                (workflow_id, module),
            )
        rows = cur.fetchall()
        trajectories: Dict[str, List[Dict[str, float]]] = {}
        for mod, r_id, sr, delta, ma, var in rows:
            trajectories.setdefault(str(mod), []).append(
                {
                    "run_id": str(r_id),
                    "success_rate": float(sr),
                    "roi_delta": float(delta),
                    "moving_avg": float(ma),
                    "variance": float(var),
                }
            )
        return trajectories

    # ------------------------------------------------------------------
    def module_impact_report(
        self, workflow_id: str, run_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Return modules grouped by improvement sign for ``run_id``."""

        cur = self.conn.cursor()
        cur.execute(
            (
                "SELECT run_id, module_deltas FROM workflow_results "
                "WHERE workflow_id=? ORDER BY timestamp"
            ),
            (workflow_id,),
        )
        rows = cur.fetchall()

        prev: Dict[str, float] = {}
        for r_id, deltas_json in rows:
            deltas = json.loads(deltas_json or "{}")
            if r_id == run_id:
                improved: Dict[str, Dict[str, float]] = {}
                regressed: Dict[str, Dict[str, float]] = {}
                for mod, metrics in deltas.items():
                    curr = float(metrics.get("roi_delta", 0.0))
                    sr = float(metrics.get("success_rate", 0.0))
                    diff = curr - prev.get(mod, 0.0)
                    entry = {"roi_delta": diff, "success_rate": sr}
                    if diff >= 0:
                        improved[mod] = entry
                    else:
                        regressed[mod] = entry
                return {"improved": improved, "regressed": regressed}
            for mod, metrics in deltas.items():
                prev[mod] = float(metrics.get("roi_delta", 0.0))

        return {"improved": {}, "regressed": {}}

    def fetch_module_volatility(
        self, workflow_id: str, module: str
    ) -> Dict[str, float]:
        """Return latest moving average and variance for ``module``.

        The metrics are read from the most recent entry in the
        ``workflow_module_deltas`` table and correspond to the ``moving_avg``
        and ``variance`` fields provided by
        :meth:`fetch_module_trajectories`.
        """

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT roi_delta_ma, roi_delta_var FROM workflow_module_deltas
            WHERE workflow_id=? AND module=?
            ORDER BY id DESC LIMIT 1
            """,
            (workflow_id, module),
        )
        row = cur.fetchone()
        if row:
            return {"moving_avg": float(row[0]), "variance": float(row[1])}
        return {"moving_avg": 0.0, "variance": 0.0}


def module_impact_report(
    workflow_id: str, run_id: str, db_path: str | Path = _ROI_RESULTS_DB_PATH
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper returning module impact report from ``db_path``."""

    db = ROIResultsDB(db_path)
    return db.module_impact_report(workflow_id, run_id)


def module_performance_trajectories(
    workflow_id: str,
    module: str | None = None,
    db_path: str | Path = _ROI_RESULTS_DB_PATH,
) -> Dict[str, List[Dict[str, float]]]:
    """Convenience wrapper returning module trend data from ``db_path``.

    The returned structure mirrors
    :meth:`ROIResultsDB.fetch_module_trajectories` and therefore includes
    per-run moving averages and variance for each module.
    """

    db = ROIResultsDB(db_path)
    return db.fetch_module_trajectories(workflow_id, module)


def module_volatility(
    workflow_id: str,
    module: str,
    db_path: str | Path = _ROI_RESULTS_DB_PATH,
) -> Dict[str, float]:
    """Convenience wrapper returning latest volatility metrics."""

    db = ROIResultsDB(db_path)
    return db.fetch_module_volatility(workflow_id, module)


def workflow_trends(
    workflow_id: str, db_path: str | Path = _ROI_RESULTS_DB_PATH
) -> List[Dict[str, float]]:
    """Convenience wrapper returning aggregate workflow trends from ``db_path``."""

    db = ROIResultsDB(db_path)
    return db.fetch_trends(workflow_id)


def compute_rolling_metrics(
    trends: List[Dict[str, float]], window: int = 5
) -> List[Dict[str, float]]:
    """Append rolling average and slope for each metric in ``trends``.

    Returns a new list of dictionaries where each item includes additional
    ``<metric>_avg`` and ``<metric>_slope`` keys based on the preceding ``window``
    entries (including the current one).  Slope is calculated as the difference
    between the first and last values over the window divided by the number of
    steps.
    """

    def _rolling(values: List[float]) -> List[tuple[float, float]]:
        stats: List[tuple[float, float]] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            segment = values[start:i + 1]
            avg = sum(segment) / len(segment)
            if len(segment) > 1:
                slope = (segment[-1] - segment[0]) / (len(segment) - 1)
            else:
                slope = 0.0
            stats.append((avg, slope))
        return stats

    metrics = [
        "roi_gain",
        "workflow_synergy_score",
        "bottleneck_index",
        "patchability_score",
    ]
    series: Dict[str, List[float]] = {m: [t[m] for t in trends] for m in metrics}
    rolling: Dict[str, List[tuple[float, float]]] = {
        m: _rolling(vals) for m, vals in series.items()
    }
    out: List[Dict[str, float]] = []
    for i, base in enumerate(trends):
        entry = dict(base)
        for m in metrics:
            avg, slope = rolling[m][i]
            entry[f"{m}_avg"] = avg
            entry[f"{m}_slope"] = slope
        out.append(entry)
    return out


__all__ = [
    "ROIResult",
    "ROIResultsDB",
    "module_impact_report",
    "module_performance_trajectories",
    "module_volatility",
    "workflow_trends",
    "compute_rolling_metrics",
]
