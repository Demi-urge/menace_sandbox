from __future__ import annotations

"""ROI scoring utilities for workflow evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import sqlite3
import uuid
from itertools import combinations
from statistics import stdev

import numpy as np

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator
from .data_bot import MetricsDB
from .neuroplasticity import PathwayDB
try:  # ensure telemetry backend uses same router
    from . import telemetry_backend as tb
except Exception:  # pragma: no cover - optional
    tb = None  # type: ignore


@dataclass
class Scorecard:
    """Structured result for a single workflow run."""

    workflow_id: str
    runtime: float
    success: bool
    roi_gain: float
    metrics: Dict[str, float]
    workflow_synergy: float = 0.0
    bottleneck_index: float = 0.0
    patchability: float = 0.0


LOCAL_TABLES.add("workflow_results")
LOCAL_TABLES.add("workflow_module_deltas")
router = GLOBAL_ROUTER or init_db_router("roi_scorer")
if tb is not None:  # align telemetry router
    tb.GLOBAL_ROUTER = router


def compute_workflow_synergy(mod_metrics: Dict[str, list[float]]) -> float:
    """Return average pairwise correlation across module ROI histories.

    ``mod_metrics`` should map module names to lists of ROI deltas or scores
    over time.  The helper computes the Pearson correlation for every pair of
    modules and returns their mean.  If insufficient data is provided the
    function returns ``0.0``.
    """

    arrays = [np.asarray(v, dtype=float) for v in mod_metrics.values() if len(v) >= 2]
    if len(arrays) < 2:
        return 0.0
    total = 0.0
    count = 0
    for a, b in combinations(arrays, 2):
        if a.std() == 0 or b.std() == 0:
            continue
        total += float(np.corrcoef(a, b)[0, 1])
        count += 1
    return total / count if count else 0.0


def compute_bottleneck_index(timings: Dict[str, Tuple[float, float]]) -> float:
    """Return bottleneck ratio adjusted by failure frequency.

    ``timings`` maps module name to a tuple of ``(runtime, failures)``.  The
    ratio of the slowest runtime to total runtime is scaled by how frequently
    that module fails relative to the whole workflow.  Missing or empty data
    yields ``0.0``.
    """

    if not timings:
        return 0.0
    total_runtime = sum(t for t, _ in timings.values())
    if total_runtime <= 0:
        return 0.0
    slowest_mod, (slowest_rt, slowest_fail) = max(
        timings.items(), key=lambda x: x[1][0]
    )
    total_fail = sum(f for _, f in timings.values())
    fail_freq = slowest_fail / total_fail if total_fail > 0 else 0.0
    return (slowest_rt / total_runtime) * (1 + fail_freq)


def compute_patchability(history: list[float]) -> float:
    """Return patchability score from ROI history.

    The metric multiplies the slope of ROI improvement by ``1/(sigma + 1)``
    where ``sigma`` is the standard deviation of ``history``.  A positive slope
    with low volatility yields higher patchability.
    """

    if not history or len(history) < 2:
        return 0.0
    slope = (history[-1] - history[0]) / (len(history) - 1)
    try:
        sigma = stdev(history)
    except Exception:
        sigma = 0.0
    return slope * (1.0 / (sigma + 1.0))


def insert_workflow_result(
    cur: sqlite3.Cursor,
    workflow_id: str,
    run_id: str,
    runtime: float,
    success: bool,
    roi_gain: float,
    workflow_synergy: float,
    bottleneck_index: float,
    patchability: float,
) -> None:
    """Insert a workflow result into ``workflow_results``.

    Parameters
    ----------
    cur:
        Database cursor targeting ``workflow_results``.
    workflow_id:
        Identifier of the evaluated workflow.
    run_id:
        Generated run identifier.
    runtime, success, roi_gain, workflow_synergy, bottleneck_index, patchability:
        Recorded metrics for the workflow execution.
    """

    cur.execute(
        """
        INSERT INTO workflow_results(
            workflow_id, run_id, runtime, success, roi_gain,
            workflow_synergy_score, bottleneck_index, patchability_score
        ) VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            workflow_id,
            run_id,
            runtime,
            int(success),
            roi_gain,
            workflow_synergy,
            bottleneck_index,
            patchability,
        ),
    )


class ROIScorer:
    """Base scorer storing ROI results in ``roi_results.db``."""

    def __init__(
        self,
        db_path: str | Path = "roi_results.db",
        tracker: ROITracker | None = None,
        calculator: ROICalculator | None = None,
        profile_type: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.tracker = tracker or ROITracker()
        self.calculator = calculator or ROICalculator()
        self.profile_type = profile_type or next(iter(self.calculator.profiles))
        self._init_db()

    # internal
    def _init_db(self) -> None:
        self._db = DBRouter('roi_results', str(self.db_path), str(self.db_path))
        self.conn = self._db.get_connection('workflow_results', operation='write')
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_results(
                id INTEGER PRIMARY KEY,
                workflow_id TEXT,
                run_id TEXT,
                runtime REAL,
                success INTEGER,
                roi_gain REAL,
                workflow_synergy_score REAL,
                bottleneck_index REAL,
                patchability_score REAL,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                roi_delta REAL,
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
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roi_results'"
        )
        if cur.fetchone():
            cur.execute(
                """
                INSERT INTO workflow_results(
                    workflow_id, run_id, runtime, success, roi_gain,
                    workflow_synergy_score, bottleneck_index, patchability_score, ts
                )
                SELECT
                    workflow_id,
                    CAST(run_id AS TEXT),
                    runtime,
                    success,
                    roi_gain,
                    workflow_synergy,
                    bottleneck_index,
                    patchability,
                    datetime(ts, 'unixepoch')
                FROM roi_results
                """,
            )
            cur.execute("DROP TABLE roi_results")
        self.conn.commit()

    def _persist(
        self,
        workflow_id: str,
        runtime: float,
        success: bool,
        roi_gain: float,
        workflow_synergy: float,
        bottleneck_index: float,
        patchability: float,
    ) -> str:
        cur = self.conn.cursor()
        run_id = uuid.uuid4().hex
        insert_workflow_result(
            cur,
            workflow_id,
            run_id,
            runtime,
            success,
            roi_gain,
            workflow_synergy,
            bottleneck_index,
            patchability,
        )
        self.conn.commit()
        return run_id


class CompositeWorkflowScorer(ROIScorer):
    """Run complete workflows and record ROI metrics."""

    def __init__(
        self,
        metrics_db: MetricsDB,
        pathway_db: PathwayDB,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metrics_db = metrics_db
        self.pathway_db = pathway_db
        self._module_offsets: Dict[str, int] = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        self._last_module_deltas: Dict[str, float] = {}

    def score(
        self, workflow_id: str, workflow_callable: Callable[[], bool]
    ) -> Tuple[str, Scorecard]:
        """Execute *workflow_callable* and persist scoring information."""
        start_counts = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        try:
            prev_rows = self.metrics_db.fetch_eval(workflow_id)
        except sqlite3.ProgrammingError:
            global router
            router = init_db_router("roi_scorer")
            if tb is not None:
                tb.GLOBAL_ROUTER = router
            self.metrics_db.router = router
            self.pathway_db.router = router
            prev_rows = self.metrics_db.fetch_eval(workflow_id)
        from .workflow_benchmark import benchmark_workflow  # local import to avoid circular

        success = benchmark_workflow(
            workflow_callable, self.metrics_db, self.pathway_db, name=workflow_id
        )
        new_rows = self.metrics_db.fetch_eval(workflow_id)[len(prev_rows):]
        metrics: Dict[str, float] = {}
        for _, metric, value, _ in new_rows:
            try:
                metrics[metric] = float(value)
            except Exception:
                continue
        runtime = metrics.get("duration", 0.0)
        roi_before = 0.0
        calc_metrics = {
            "reliability": 1.0 if success else 0.0,
            "efficiency": 1.0 / runtime if runtime > 0 else 0.0,
        }
        roi_after, _, _ = self.calculator.calculate(calc_metrics, self.profile_type)
        roi_gain = roi_after - roi_before

        workflow_synergy = compute_workflow_synergy(self.tracker.metrics_history)

        timings: Dict[str, Tuple[float, float]] = {}
        for name, val in metrics.items():
            if name.endswith("_runtime") or name.endswith("_time"):
                base = name.rsplit("_", 1)[0]
                failures = float(
                    metrics.get(f"{base}_failures", metrics.get(f"{base}_failure_rate", 0.0))
                )
                timings[base] = (float(val), failures)
        bottleneck_index = compute_bottleneck_index(timings)

        patchability = compute_patchability(self.tracker.roi_history)

        metrics["workflow_synergy"] = workflow_synergy
        metrics["bottleneck_index"] = bottleneck_index
        metrics["patchability"] = patchability

        run_id = self._persist(
            workflow_id,
            runtime,
            success,
            roi_gain,
            workflow_synergy,
            bottleneck_index,
            patchability,
        )
        mod_deltas: Dict[str, float] = {}
        for mod, deltas in self.tracker.module_deltas.items():
            start = start_counts.get(mod, 0)
            if start < len(deltas):
                mod_deltas[mod] = sum(float(x) for x in deltas[start:])
        self._last_module_deltas = mod_deltas
        self._module_offsets = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        cur = self.conn.cursor()
        for mod in set(mod_deltas) | set(timings):
            runtime_mod = timings.get(mod, (0.0, 0.0))[0]
            delta = mod_deltas.get(mod, 0.0)
            cur.execute(
                """
                INSERT INTO workflow_module_deltas(
                    workflow_id, run_id, module, runtime, roi_delta
                ) VALUES(?,?,?,?,?)
                """,
                (workflow_id, run_id, mod, runtime_mod, delta),
            )
        self.conn.commit()
        for key, value in metrics.items():
            self.tracker.metrics_history.setdefault(key, []).append(value)
        scorecard = Scorecard(
            workflow_id,
            runtime,
            success,
            roi_gain,
            metrics,
            workflow_synergy,
            bottleneck_index,
            patchability,
        )
        return run_id, scorecard

    def module_deltas(self) -> Dict[str, float]:
        """Return ROI contribution per module from the last ``score`` call."""
        return dict(self._last_module_deltas)


__all__ = [
    "ROIScorer",
    "CompositeWorkflowScorer",
    "Scorecard",
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
    "insert_workflow_result",
]
