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
from collections import defaultdict, deque

import numpy as np

from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator
from .roi_results_db import ROIResultsDB
from . import sandbox_runner


# ---------------------------------------------------------------------------
# Helper metrics copied from ``roi_scorer`` to avoid heavy dependencies.
# ---------------------------------------------------------------------------


def compute_workflow_synergy(
    roi_history: Iterable[float],
    module_history: Mapping[str, Iterable[float]],
    window: int = 5,
) -> float:
    """Mean correlation between module ROI deltas and overall ROI.

    The score examines the ``window`` most recent ROI values along with the
    corresponding per-module ROI deltas.  For each module a Pearson
    correlation coefficient with the overall ROI history is computed and the
    average of all valid correlations is returned.  Modules with insufficient
    variance or history are ignored.  A value close to ``1.0`` indicates strong
    positive alignment between modules and overall ROI while ``0.0`` denotes no
    discernible relationship.
    """

    roi_list = list(roi_history)[-window:]
    if len(roi_list) < 2 or not module_history:
        return 0.0
    correlations: list[float] = []
    for deltas in module_history.values():
        mod_list = list(deltas)[-window:]
        if len(mod_list) != len(roi_list):
            continue
        if np.std(mod_list) == 0 or np.std(roi_list) == 0:
            continue
        corr = float(np.corrcoef(roi_list, mod_list)[0, 1])
        correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


def compute_bottleneck_index(tracker: ROITracker, _workflow_id: str | None = None) -> float:
    """Root mean square deviation of normalised module runtimes.

    The metric first normalises module runtimes so they sum to one.  It then
    computes the root mean square deviation from an ideal uniform distribution
    where all modules consume equal runtime.  Higher values therefore indicate
    a more pronounced bottleneck.
    """

    runtimes: Mapping[str, float] = getattr(tracker, "timings", {})
    if not runtimes:
        return 0.0
    total = sum(runtimes.values())
    if total <= 0:
        return 0.0
    norm = np.array(list(runtimes.values()), dtype=float) / total
    uniform = 1.0 / len(norm)
    return float(np.sqrt(np.mean((norm - uniform) ** 2)))


def compute_patchability(history: Iterable[float], window: int = 5) -> float:
    """ROI trend slope normalised by recent volatility.

    A simple linear regression is fit to the last ``window`` ROI values to
    obtain the improvement slope.  This slope is divided by the standard
    deviation of the same window, yielding a volatility-adjusted indicator of
    how patchable the workflow appears.  Higher values indicate a consistent
    upward trend with low variance.
    """

    hist_list = list(history)[-window:]
    if len(hist_list) < 2:
        return 0.0
    x = np.arange(len(hist_list))
    slope = float(np.polyfit(x, hist_list, 1)[0])
    volatility = float(np.std(hist_list))
    if volatility == 0:
        return 0.0
    return slope / volatility


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
        self.history_window = 20
        self._module_roi_history: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.history_window)
        )
        self.module_attribution: Dict[str, Dict[str, float]] = {}

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

        per_module: Dict[str, Dict[str, float]] = {}
        timings = getattr(self.tracker, "timings", {})
        total_runtime = sum(float(t) for t in timings.values())
        for mod, deltas in self.tracker.module_deltas.items():
            start_idx = self._module_start.get(mod, 0)
            roi_delta = sum(float(x) for x in deltas[start_idx:])
            runtime = float(timings.get(mod, 0.0))
            bottleneck = runtime / total_runtime if total_runtime > 0 else 0.0
            per_module[mod] = {
                "runtime": runtime,
                "roi_delta": roi_delta,
                "success_rate": 1.0 if self._module_successes.get(mod) else 0.0,
                "bottleneck_contribution": bottleneck,
            }
            self._module_roi_history[mod].append(roi_delta)
            self.results_db.log_module_attribution(mod, roi_delta, bottleneck)
        self.module_attribution = {
            mod: {
                "roi_delta": data["roi_delta"],
                "bottleneck_contribution": data["bottleneck_contribution"],
            }
            for mod, data in per_module.items()
        }

        workflow_synergy_score = compute_workflow_synergy(
            self.tracker.roi_history, self._module_roi_history, self.history_window
        )
        bottleneck_index = compute_bottleneck_index(self.tracker, workflow_id)
        patchability_score = compute_patchability(
            self.tracker.roi_history, self.history_window
        )

        self.results_db.log_result(
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

        per_module_metrics: Dict[str, Dict[str, float]] = {}
        timings = getattr(tracker, "timings", {})
        total_runtime = sum(float(t) for t in timings.values())
        for mod, counts in per_module_counts.items():
            roi_delta = sum(float(x) for x in tracker.module_deltas.get(mod, []))
            sr = counts["success"] / counts["total"] if counts["total"] else 0.0
            runtime = float(timings.get(mod, 0.0))
            bottleneck = runtime / total_runtime if total_runtime > 0 else 0.0
            per_module_metrics[mod] = {
                "success_rate": sr,
                "roi_delta": roi_delta,
                "runtime": runtime,
                "bottleneck_contribution": bottleneck,
            }
            self._module_roi_history[mod].append(roi_delta)
            self.results_db.log_module_attribution(mod, roi_delta, bottleneck)
        self.module_attribution = {
            mod: {
                "roi_delta": data["roi_delta"],
                "bottleneck_contribution": data["bottleneck_contribution"],
            }
            for mod, data in per_module_metrics.items()
        }

        workflow_synergy_score = compute_workflow_synergy(
            getattr(tracker, "roi_history", []), self._module_roi_history, self.history_window
        )
        bottleneck_index = compute_bottleneck_index(tracker, workflow_id)
        patchability_score = compute_patchability(
            getattr(tracker, "roi_history", []), self.history_window
        )

        run_id = uuid.uuid4().hex
        self.results_db.log_result(
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

