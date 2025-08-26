"""Lightweight workflow scoring utilities for self-improvement modules.

This module provides a trimmed down :class:`CompositeWorkflowScorer` that
executes a workflow callable, records ROI related metrics via
``ROITracker``/``ROICalculator`` and persists aggregated results in
``ROIResultsDB``.  The implementation mirrors the behaviour of the heavier
``roi_scorer`` module but keeps the dependency surface small so that unit
tests can exercise the scoring pipeline in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple
import time
import uuid

import numpy as np

from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator
from .roi_results_db import ROIResultsDB
from . import sandbox_runner


# ---------------------------------------------------------------------------
# Helper metrics copied from ``roi_scorer`` to avoid heavy dependencies.
# ---------------------------------------------------------------------------


def compute_workflow_synergy(
    tracker: ROITracker, modules: Iterable[str] | None = None
) -> float:
    """Aggregate pairwise synergy metrics for ``modules``."""

    mods = {str(m) for m in (modules if modules is not None else tracker.module_deltas)}
    try:
        from . import synergy_history_db as shd  # type: ignore

        history = shd.load_history()
        vals: list[float] = []
        for entry in history:
            for pair, score in entry.items():
                if not isinstance(score, (int, float)):
                    continue
                parts = str(pair).split(",")
                if len(parts) == 2 and {parts[0], parts[1]} <= mods:
                    vals.append(float(score))
        if vals:
            return float(sum(vals) / len(vals))
    except Exception:
        pass

    try:  # pragma: no cover - best effort
        from .sandbox_runner.environment import aggregate_synergy_metrics

        results = aggregate_synergy_metrics(list(mods))
        if results:
            return float(sum(val for _, val in results) / len(results))
    except Exception:
        pass
    return 0.0


def compute_bottleneck_index(tracker: ROITracker, workflow_id: str) -> float:
    """Return worst runtime-to-ROI ratio adjusted by workflow variance."""

    runtimes: Mapping[str, float] = getattr(tracker, "timings", {})
    ratios: Dict[str, float] = {}
    for mod, runtime in runtimes.items():
        roi = sum(float(x) for x in tracker.module_deltas.get(mod, []))
        if roi == 0:
            roi = 1e-9
        ratios[mod] = float(runtime) / roi
    if not ratios:
        return 0.0
    worst_ratio = max(ratios.values())
    variance = 0.0
    if hasattr(tracker, "workflow_variance"):
        try:
            variance = float(tracker.workflow_variance(workflow_id))
        except Exception:
            variance = 0.0
    return worst_ratio * (1.0 + variance)


def compute_patchability(history: Iterable[float], window: int = 5) -> float:
    """Return ROI slope over recent runs using linear regression."""

    hist_list = list(history)[-window:]
    if len(hist_list) < 2:
        return 0.0
    x = np.arange(len(hist_list))
    slope = float(np.polyfit(x, hist_list, 1)[0])
    return slope


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Structured outcome of a workflow evaluation."""

    runtime: float
    success_rate: float
    roi_gain: float
    workflow_synergy_score: float
    bottleneck_index: float
    patchability_score: float
    per_module: Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class ROIScorer:
    """Thin wrapper around :class:`ROITracker` and ``ROICalculator``."""

    def __init__(
        self,
        tracker: ROITracker | None = None,
        calculator: ROICalculator | None = None,
        profile_type: str | None = None,
    ) -> None:
        self.tracker = tracker or ROITracker()
        if calculator is not None:
            self.calculator = calculator
        else:
            try:
                self.calculator = ROICalculator()
            except Exception:
                # Fall back to a tiny stub that sums provided metrics.  This keeps
                # unit tests lightweight when the full ROI profile configuration is
                # unavailable.
                class _StubCalc:
                    profiles = {"default": {}}

                    def calculate(self, metrics: Dict[str, Any], _profile: str) -> Tuple[float, bool, list[str]]:
                        return float(sum(float(v) for v in metrics.values())), False, []

                self.calculator = _StubCalc()
        self.profile_type = profile_type or next(iter(self.calculator.profiles))


class CompositeWorkflowScorer(ROIScorer):
    """Execute workflows and compute aggregate ROI metrics."""

    def __init__(
        self,
        tracker: ROITracker | None = None,
        calculator: ROICalculator | None = None,
        results_db: ROIResultsDB | None = None,
        profile_type: str | None = None,
    ) -> None:
        super().__init__(tracker, calculator, profile_type)
        self.results_db = results_db or ROIResultsDB()
        # offsets used to compute deltas for individual runs
        self._roi_start: int = 0
        self._module_start: Dict[str, int] = {}
        self._module_successes: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    def run(
        self,
        workflow_callable: Callable[[], bool],
        workflow_id: str,
        run_id: str,
    ) -> EvaluationResult:
        """Execute ``workflow_callable`` and persist aggregated metrics."""

        start = time.perf_counter()
        try:
            success = bool(workflow_callable())
        except Exception:
            success = False
        runtime = time.perf_counter() - start

        roi_gain = sum(
            float(x) for x in self.tracker.roi_history[self._roi_start :]
        )
        modules = list(self.tracker.module_deltas)
        workflow_synergy_score = compute_workflow_synergy(self.tracker, modules)
        bottleneck_index = compute_bottleneck_index(self.tracker, workflow_id)
        patchability_score = compute_patchability(self.tracker.roi_history)

        per_module: Dict[str, Dict[str, float]] = {}
        timings = getattr(self.tracker, "timings", {})
        for mod, deltas in self.tracker.module_deltas.items():
            start_idx = self._module_start.get(mod, 0)
            roi_delta = sum(float(x) for x in deltas[start_idx:])
            per_module[mod] = {
                "runtime": float(timings.get(mod, 0.0)),
                "roi_delta": roi_delta,
                "success_rate": 1.0 if self._module_successes.get(mod) else 0.0,
            }

        self.results_db.add_result(
            workflow_id=workflow_id,
            run_id=run_id,
            runtime=runtime,
            success_rate=1.0 if success else 0.0,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            module_deltas=per_module,
        )

        return EvaluationResult(
            runtime=runtime,
            success_rate=1.0 if success else 0.0,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            per_module=per_module,
        )

    # ------------------------------------------------------------------
    def score_workflow(
        self,
        workflow_id: str,
        modules: Mapping[str, Callable[[], bool]],
        run_id: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute ``modules`` sequentially and return aggregate metrics."""

        run_id = run_id or uuid.uuid4().hex
        self._roi_start = len(self.tracker.roi_history)
        self._module_start = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        self._module_successes = {}
        self.tracker.timings = {}

        def workflow_callable() -> bool:
            overall = True
            for name, func in modules.items():
                t0 = time.perf_counter()
                try:
                    ok = bool(func())
                except Exception:
                    ok = False
                duration = time.perf_counter() - t0
                self.tracker.timings[name] = duration
                self._module_successes[name] = ok
                metrics = {
                    "reliability": 1.0 if ok else 0.0,
                    "efficiency": 1.0 / duration if duration > 0 else 0.0,
                }
                roi_after, _, _ = self.calculator.calculate(
                    metrics, self.profile_type
                )
                self.tracker.update(
                    0.0,
                    roi_after,
                    modules=[name],
                    metrics=metrics,
                    profile_type=self.profile_type,
                )
                overall = overall and ok
            return overall

        result = self.run(workflow_callable, workflow_id, run_id)
        return run_id, asdict(result)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        workflow_id: str,
        env_presets: Mapping[str, list[Dict[str, Any]]] | list[Dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        """Backwards compatible wrapper executing sandbox simulations."""

        start = time.perf_counter()
        tracker, details = sandbox_runner.environment.run_workflow_simulations(
            workflows_db=workflow_id,
            env_presets=env_presets,
            return_details=True,
            tracker=self.tracker,
        )
        runtime = time.perf_counter() - start

        total = success = 0
        per_module_counts: Dict[str, Dict[str, int]] = {}
        for runs in details.values():
            for entry in runs:
                result = entry.get("result", {})
                module = entry.get("module")
                total += 1
                if result.get("exit_code", 1) == 0:
                    success += 1
                    if module:
                        per_module_counts.setdefault(module, {"total": 0, "success": 0})
                        per_module_counts[module]["success"] += 1
                if module:
                    per_module_counts.setdefault(module, {"total": 0, "success": 0})
                    per_module_counts[module]["total"] += 1

        success_rate = success / total if total else 0.0
        roi_gain = sum(float(r) for r in getattr(tracker, "roi_history", []))
        modules = list(per_module_counts)
        workflow_synergy_score = compute_workflow_synergy(tracker, modules)
        bottleneck_index = compute_bottleneck_index(tracker, workflow_id)
        patchability_score = compute_patchability(getattr(tracker, "roi_history", []))

        per_module_metrics: Dict[str, Dict[str, float]] = {}
        for mod, counts in per_module_counts.items():
            roi_delta = sum(float(x) for x in tracker.module_deltas.get(mod, []))
            sr = counts["success"] / counts["total"] if counts["total"] else 0.0
            per_module_metrics[mod] = {"success_rate": sr, "roi_delta": roi_delta}

        run_id = uuid.uuid4().hex
        self.results_db.add_result(
            workflow_id=workflow_id,
            run_id=run_id,
            runtime=runtime,
            success_rate=success_rate,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            module_deltas=per_module_metrics,
        )

        return EvaluationResult(
            runtime=runtime,
            success_rate=success_rate,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            per_module=per_module_metrics,
        )


__all__ = [
    "ROIScorer",
    "CompositeWorkflowScorer",
    "EvaluationResult",
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
]

