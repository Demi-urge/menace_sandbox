from __future__ import annotations

"""ROI scoring utilities for workflow evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import time
import sqlite3

from db_router import GLOBAL_ROUTER, init_db_router, LOCAL_TABLES
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


LOCAL_TABLES.add("roi_results")
router = GLOBAL_ROUTER or init_db_router("roi_scorer")
if tb is not None:  # align telemetry router
    tb.GLOBAL_ROUTER = router


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
        conn = router.get_connection("roi_results", operation="write")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roi_results (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                runtime REAL NOT NULL,
                success INTEGER NOT NULL,
                roi_gain REAL NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        conn.commit()

    def _persist(self, workflow_id: str, runtime: float, success: bool, roi_gain: float) -> int:
        global router
        conn = router.get_connection("roi_results", operation="write")
        try:
            cur = conn.cursor()
        except sqlite3.ProgrammingError:
            router = init_db_router("roi_scorer")
            if tb is not None:
                tb.GLOBAL_ROUTER = router
            conn = router.get_connection("roi_results", operation="write")
            cur = conn.cursor()
        cur.execute(
            "INSERT INTO roi_results (workflow_id, runtime, success, roi_gain, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (workflow_id, runtime, int(success), roi_gain, time.time()),
        )
        conn.commit()
        return int(cur.lastrowid)


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

    def score(
        self, workflow_id: str, workflow_callable: Callable[[], bool]
    ) -> Tuple[int, Scorecard]:
        """Execute *workflow_callable* and persist scoring information."""

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
        run_id = self._persist(workflow_id, runtime, success, roi_gain)
        for key, value in metrics.items():
            self.tracker.metrics_history.setdefault(key, []).append(value)
        scorecard = Scorecard(workflow_id, runtime, success, roi_gain, metrics)
        return run_id, scorecard


__all__ = ["ROIScorer", "CompositeWorkflowScorer", "Scorecard"]
