from __future__ import annotations

"""Lightweight workflow scoring utilities for sandbox evaluation."""

from dataclasses import dataclass
from statistics import stdev
from typing import Any, Dict, Mapping, Iterable
import time
import uuid

import numpy as np

from .roi_tracker import ROITracker
from . import sandbox_runner
from .roi_results_db import ROIResultsDB


# ---------------------------------------------------------------------------
# Metric helpers copied from ``roi_scorer`` to avoid heavy dependencies.
# ---------------------------------------------------------------------------

def compute_workflow_synergy(tracker: ROITracker, window: int = 5) -> float:
    """Return average of recent ``synergy_*`` metrics."""

    eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
    rel_hist = tracker.metrics_history.get("synergy_reliability", [])
    eff_avg = sum(eff_hist[-window:]) / len(eff_hist[-window:]) if eff_hist else None
    rel_avg = sum(rel_hist[-window:]) / len(rel_hist[-window:]) if rel_hist else None
    vals = [v for v in (eff_avg, rel_avg) if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_bottleneck_index(timings: Mapping[str, float]) -> float:
    """Return proportion of total latency from the slowest module."""

    if not timings:
        return 0.0
    total_runtime = sum(timings.values())
    if total_runtime <= 0:
        return 0.0
    slowest_rt = max(timings.values())
    return slowest_rt / total_runtime


def compute_patchability(history: Iterable[float], patch_success: float = 1.0) -> float:
    """Return patchability score from ROI history and patch success rate."""

    hist_list = list(history)
    if len(hist_list) < 2:
        return 0.0
    x = np.arange(len(hist_list))
    slope = float(np.polyfit(x, hist_list, 1)[0])
    try:
        sigma = stdev(hist_list)
    except Exception:
        sigma = 0.0
    return slope * (1.0 / (sigma + 1.0)) * float(patch_success)


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
    """Thin wrapper around :class:`ROITracker`."""

    def __init__(self, tracker: ROITracker | None = None) -> None:
        self.tracker = tracker or ROITracker()


class CompositeWorkflowScorer(ROIScorer):
    """Execute workflows and compute aggregate ROI metrics."""

    def __init__(self, tracker: ROITracker | None = None, results_db: ROIResultsDB | None = None) -> None:
        super().__init__(tracker)
        self.results_db = results_db or ROIResultsDB()

    def evaluate(
        self,
        workflow_id: str,
        env_presets: Mapping[str, list[Dict[str, Any]]] | list[Dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        """Run ``workflow_id`` under ``env_presets`` and summarise metrics."""

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
        workflow_synergy_score = compute_workflow_synergy(tracker)
        bottleneck_index = compute_bottleneck_index(getattr(tracker, "timings", {}))
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
