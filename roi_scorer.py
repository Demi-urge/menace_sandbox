from __future__ import annotations

"""ROI scoring utilities for workflow evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple
import sqlite3
import uuid
import statistics

from .workflow_metrics import (
    compute_bottleneck_index,
    compute_patchability,
    compute_workflow_synergy,
)

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator
from .data_bot import MetricsDB
from .neuroplasticity import PathwayDB
from .roi_results_db import ROIResultsDB
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
    workflow_synergy_score: float = 0.0
    bottleneck_index: float = 0.0
    patchability_score: float = 0.0


LOCAL_TABLES.add("workflow_results")
LOCAL_TABLES.add("workflow_module_deltas")
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
        self._db = DBRouter('workflow_results', str(self.db_path), str(self.db_path))
        self.conn = self._db.get_connection('workflow_results', operation='write')
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
            CREATE TABLE IF NOT EXISTS workflow_module_deltas(
                id INTEGER PRIMARY KEY,
                workflow_id TEXT,
                run_id TEXT,
                module TEXT,
                runtime REAL,
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
            r[1] for r in cur.execute("PRAGMA table_info(workflow_module_deltas)").fetchall()
        }
        if "roi_delta_ma" not in existing:
            cur.execute(
                "ALTER TABLE workflow_module_deltas ADD COLUMN roi_delta_ma REAL DEFAULT 0.0"
            )
        if "roi_delta_var" not in existing:
            cur.execute(
                "ALTER TABLE workflow_module_deltas ADD COLUMN roi_delta_var REAL DEFAULT 0.0"
            )
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roi_results'",
        )
        if cur.fetchone():
            cur.execute(
                """
                INSERT INTO workflow_results(
                    workflow_id, run_id, timestamp, runtime, success_rate, roi_gain,
                    workflow_synergy_score, bottleneck_index, patchability_score, module_deltas
                )
                SELECT
                    workflow_id,
                    CAST(run_id AS TEXT),
                    datetime(ts, 'unixepoch'),
                    runtime,
                    success,
                    roi_gain,
                    workflow_synergy,
                    bottleneck_index,
                    patchability,
                    '{}'
                FROM roi_results
                """,
            )
            cur.execute("DROP TABLE roi_results")
        self.conn.commit()


class CompositeWorkflowScorer(ROIScorer):
    """Run complete workflows and record ROI metrics."""

    def __init__(
        self,
        metrics_db: MetricsDB,
        pathway_db: PathwayDB,
        results_db: ROIResultsDB | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metrics_db = metrics_db
        self.pathway_db = pathway_db
        self.results_db = results_db or ROIResultsDB(self.db_path, router=self._db)
        self._module_offsets: Dict[str, int] = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        self._last_module_deltas: Dict[str, float] = {}

    def score_workflow(
        self,
        workflow_id: str,
        modules: Mapping[str, Callable[[], bool]],
        run_id: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute ``modules`` sequentially and persist scoring information."""
        start_counts = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        baseline_rois: Dict[str, float] = {}
        for mod in modules:
            runs = self.results_db.fetch_results(workflow_id)
            if runs:
                last = runs[-1].module_deltas.get(mod, {})
                baseline_rois[mod] = float(last.get("roi_delta", 0.0))
            else:
                baseline_rois[mod] = 0.0
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
        from .workflow_benchmark import benchmark_workflow  # local import

        def workflow_callable() -> bool:
            overall = True
            for func in modules.values():
                try:
                    overall = bool(func()) and overall
                except Exception:
                    overall = False
            return overall

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

        workflow_synergy_score = compute_workflow_synergy(
            self.tracker.roi_history, self.tracker.module_deltas
        )

        timings: Dict[str, float] = {}
        failures: Dict[str, float] = {}
        for name, val in metrics.items():
            if name.endswith("_runtime") or name.endswith("_time"):
                base = name.rsplit("_", 1)[0]
                timings[base] = float(val)
            elif name.endswith("_failures") or name.endswith("_failure_rate"):
                base = name.rsplit("_", 1)[0]
                failures[base] = float(val)
        bottleneck_index = compute_bottleneck_index(timings)

        patch_success = 1.0
        try:  # pragma: no cover - best effort
            from .code_database import PatchHistoryDB

            patch_success = PatchHistoryDB().success_rate()
        except Exception:
            pass
        patchability_score = compute_patchability(
            self.tracker.roi_history, patch_success=patch_success
        )

        metrics["workflow_synergy_score"] = workflow_synergy_score
        metrics["bottleneck_index"] = bottleneck_index
        metrics["patchability_score"] = patchability_score

        # Update tracker with overall workflow ROI
        self.tracker.update(
            roi_before,
            roi_after,
            metrics=calc_metrics,
            profile_type=self.profile_type,
        )

        # Compute per-module ROI and update tracker
        for mod in modules:
            runtime_mod = timings.get(mod, 0.0)
            success_rate_mod = 0.0 if failures.get(mod, 0.0) else 1.0
            mod_metrics = {
                "reliability": 1.0 if success_rate_mod else 0.0,
                "efficiency": 1.0 / runtime_mod if runtime_mod > 0 else 0.0,
            }
            roi_mod_after, _, _ = self.calculator.calculate(
                mod_metrics, self.profile_type
            )
            roi_mod_before = baseline_rois.get(mod, 0.0)
            self.tracker.update(
                roi_mod_before,
                roi_mod_after,
                modules=[mod],
                metrics=mod_metrics,
                profile_type=self.profile_type,
            )

        run_id = run_id or uuid.uuid4().hex
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
            runtime_mod = timings.get(mod, 0.0)
            delta = mod_deltas.get(mod, 0.0)
            vals = self.tracker.module_deltas.get(mod, [])
            ma = float(statistics.fmean(vals)) if vals else 0.0
            var = float(statistics.pvariance(vals)) if len(vals) > 1 else 0.0
            cur.execute(
                """
                INSERT INTO workflow_module_deltas(
                    workflow_id, run_id, module, runtime, roi_delta, roi_delta_ma, roi_delta_var
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (workflow_id, run_id, mod, runtime_mod, delta, ma, var),
            )
        self.conn.commit()
        # Record aggregate workflow result with per-module deltas
        self.results_db.log_result(
            workflow_id=workflow_id,
            run_id=run_id,
            runtime=runtime,
            success_rate=1.0 if success else 0.0,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            module_deltas={m: {"roi_delta": d} for m, d in mod_deltas.items()},
        )
        for key, value in metrics.items():
            self.tracker.metrics_history.setdefault(key, []).append(value)
        result = {
            "workflow_id": workflow_id,
            "runtime": runtime,
            "success": success,
            "roi_gain": roi_gain,
            "metrics": metrics,
            "module_deltas": mod_deltas,
        }
        return run_id, result

    # backward compatibility
    def score(
        self, workflow_id: str, workflow_callable: Callable[[], bool]
    ) -> Tuple[str, Dict[str, Any]]:
        return self.score_workflow(workflow_id, {"workflow": workflow_callable})

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
]
