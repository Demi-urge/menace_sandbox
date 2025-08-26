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
import concurrent.futures
import logging

import numpy as np

from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator
from .roi_results_db import ROIResultsDB
from . import sandbox_runner
from .workflow_metrics import (
    compute_bottleneck_index,
    compute_patchability,
)


def compute_workflow_synergy(
    roi_history: Iterable[float],
    module_history: Mapping[str, Iterable[float]],
    window: int = 5,
    history_loader: Callable[[], Iterable[Dict[str, float]]] | None = None,
) -> float:
    """Weighted correlation between module ROI deltas and overall ROI.

    Historical co-performance data from ``synergy_history_db`` is used to weight
    module correlations.  ``history_loader`` can be injected to supply a custom
    loader (e.g. by tests); when not provided the loader defaults to
    :func:`synergy_history_db.load_history`.
    """

    roi_list = list(roi_history)[-window:]
    if len(roi_list) < 2 or not module_history:
        return 0.0

    if history_loader is None:
        try:
            from .synergy_history_db import load_history as history_loader
        except Exception:  # pragma: no cover - optional dependency
            history_loader = lambda: []  # type: ignore

    history = list(history_loader() or [])
    pair_totals: Dict[tuple[str, str], float] = {}
    pair_counts: Dict[tuple[str, str], int] = {}
    for entry in history:
        for key, val in entry.items():
            key = key.replace(",", "|")
            parts = [p.strip() for p in key.split("|") if p.strip()]
            if len(parts) != 2:
                continue
            a, b = parts
            pair_totals[(a, b)] = pair_totals.get((a, b), 0.0) + float(val)
            pair_totals[(b, a)] = pair_totals.get((b, a), 0.0) + float(val)
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
            pair_counts[(b, a)] = pair_counts.get((b, a), 0) + 1
    pair_avg = {p: pair_totals[p] / pair_counts[p] for p in pair_totals}

    correlations: list[float] = []
    weights: list[float] = []
    for mod, deltas in module_history.items():
        mod_list = list(deltas)[-window:]
        if len(mod_list) != len(roi_list):
            continue
        if np.std(mod_list) == 0 or np.std(roi_list) == 0:
            continue
        corr = float(np.corrcoef(roi_list, mod_list)[0, 1])
        if pair_avg:
            w_vals = [pair_avg.get((mod, other), 0.0) for other in module_history if other != mod]
            w_pos = [v for v in w_vals if v > 0]
            if not w_pos:
                continue
            weight = float(np.mean(w_pos))
        else:
            weight = 1.0
        correlations.append(corr)
        weights.append(weight)

    return float(np.average(correlations, weights=weights)) if correlations else 0.0


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
        calculator_factory: Callable[[], ROICalculator] | None = None,
        profile_type: str | None = None,
    ) -> None:
        self.tracker = tracker or ROITracker()
        try:
            self.calculator = (
                calculator_factory()
                if calculator_factory is not None
                else ROICalculator()
            )
        except Exception as exc:  # pragma: no cover - configuration errors
            raise RuntimeError(
                "ROICalculator initialization failed. Ensure ROI profile"
                " configuration is available or provide a calculator_factory"
            ) from exc
        self.profile_type = profile_type or next(iter(self.calculator.profiles))


class CompositeWorkflowScorer(ROIScorer):
    """Execute workflows and compute aggregate ROI metrics."""

    def __init__(
        self,
        tracker: ROITracker | None = None,
        calculator_factory: Callable[[], ROICalculator] | None = None,
        results_db: ROIResultsDB | None = None,
        profile_type: str | None = None,
    ) -> None:
        super().__init__(tracker, calculator_factory, profile_type)
        self.results_db = results_db or ROIResultsDB()
        # offsets used to compute deltas for individual runs
        self._roi_start: int = 0
        self._module_start: Dict[str, int] = {}
        self._module_successes: Dict[str, bool] = {}
        self._module_failures: Dict[str, str] = {}
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
        overheads = getattr(self.tracker, "scheduling_overhead", {})
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
                "scheduling_overhead": float(overheads.get(mod, 0.0)),
            }
            failure_reason = self._module_failures.get(mod)
            if failure_reason is not None:
                per_module[mod]["failure_reason"] = failure_reason
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
        bottleneck_index = compute_bottleneck_index(timings)
        patch_success = 1.0
        try:  # pragma: no cover - optional dependency
            from .code_database import PatchHistoryDB

            patch_success = PatchHistoryDB().success_rate()
        except Exception:
            pass
        patchability_score = compute_patchability(
            self.tracker.roi_history,
            window=self.history_window,
            patch_success=patch_success,
        )

        failure_reason = "; ".join(
            f"{m}: {r}" for m, r in self._module_failures.items()
        ) or None

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
            failure_reason=failure_reason,
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
        *,
        concurrency_hints: Mapping[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute ``modules`` possibly in parallel and return metrics."""

        run_id = run_id or uuid.uuid4().hex
        self._roi_start = len(self.tracker.roi_history)
        self._module_start = {
            m: len(d) for m, d in self.tracker.module_deltas.items()
        }
        self._module_successes = {}
        self._module_failures = {}
        self.tracker.timings = {}
        self.tracker.scheduling_overhead = {}

        hints = dict(concurrency_hints or {})
        max_workers = int(hints.get("max_workers", 1)) or 1

        def _run_module(name: str, func: Callable[[], bool], scheduled: float) -> bool:
            start = time.perf_counter()
            self.tracker.scheduling_overhead[name] = start - scheduled
            try:
                ok = bool(func())
            except Exception as exc:
                logging.exception("Module %s execution failed", name)
                ok = False
                self._module_failures[name] = f"{type(exc).__name__}: {exc}"
            duration = time.perf_counter() - start
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
            return ok

        def workflow_callable() -> bool:
            overall = True
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as pool:
                futures: Dict[concurrent.futures.Future[bool], str] = {}
                for name, func in modules.items():
                    scheduled = time.perf_counter() if max_workers > 1 else 0.0
                    futures[pool.submit(_run_module, name, func, scheduled)] = name
                for fut in concurrent.futures.as_completed(futures):
                    ok = bool(fut.result())
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
        overheads = getattr(tracker, "scheduling_overhead", {})
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
                "scheduling_overhead": float(overheads.get(mod, 0.0)),
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
        bottleneck_index = compute_bottleneck_index(getattr(tracker, "timings", {}))
        patch_success = 1.0
        try:  # pragma: no cover - optional dependency
            from .code_database import PatchHistoryDB

            patch_success = PatchHistoryDB().success_rate()
        except Exception:
            pass
        patchability_score = compute_patchability(
            getattr(tracker, "roi_history", []),
            window=self.history_window,
            patch_success=patch_success,
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
            failure_reason=None,
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

