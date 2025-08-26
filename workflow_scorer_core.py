from __future__ import annotations

"""Shared utilities for workflow scoring.

This module centralises metric calculations, result containers and the base
ROI scoring class used by both the lightweight :mod:`composite_workflow_scorer`
and the heavier :mod:`roi_scorer` module.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np

from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_workflow_synergy(
    roi_history: Iterable[float],
    module_history: Mapping[str, Iterable[float]],
    window: int = 5,
    history_loader: Callable[[], Iterable[Dict[str, float]]] | None = None,
) -> float:
    """Weighted correlation between module ROI deltas and overall ROI.

    Historical co-performance data from ``synergy_history_db`` is used to
    weight module correlations. ``history_loader`` can be injected to supply a
    custom loader (e.g. by tests); when not provided the loader defaults to
    :func:`synergy_history_db.load_history`.
    """

    roi_list = list(roi_history)[-window:]
    if len(roi_list) < 2 or not module_history:
        return 0.0

    if history_loader is None:
        try:
            from .synergy_history_db import load_history as history_loader
        except Exception:  # pragma: no cover - optional dependency
            def history_loader() -> Iterable[Dict[str, float]]:  # type: ignore
                return []

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
            w_vals = [
                pair_avg.get((mod, other), 0.0) for other in module_history if other != mod
            ]
            w_pos = [v for v in w_vals if v > 0]
            if not w_pos:
                continue
            weight = float(np.mean(w_pos))
        else:
            weight = 1.0
        correlations.append(corr)
        weights.append(weight)

    return float(np.average(correlations, weights=weights)) if correlations else 0.0


def compute_bottleneck_index(timings: Mapping[str, float] | Any) -> float:
    """Root mean square deviation of normalised module runtimes."""

    runtime_map: Mapping[str, float] = (
        timings if isinstance(timings, Mapping) else getattr(timings, "timings", {})
    )
    if not runtime_map:
        return 0.0
    total_runtime = sum(runtime_map.values())
    if total_runtime <= 0:
        return 0.0
    norm = np.array(list(runtime_map.values()), dtype=float) / total_runtime
    uniform = 1.0 / len(norm)
    return float(np.sqrt(np.mean((norm - uniform) ** 2)))


def compute_patchability(
    history: Iterable[float],
    window: int | None = None,
    patch_success: float = 1.0,
) -> float:
    """Volatility-adjusted ROI slope scaled by patch success rate."""

    hist_list = list(history)
    if window is not None:
        hist_list = hist_list[-window:]
    if len(hist_list) < 2:
        return 0.0
    x = np.arange(len(hist_list))
    slope = float(np.polyfit(x, hist_list, 1)[0])
    sigma = float(np.std(hist_list))
    if sigma == 0:
        return 0.0
    return slope / sigma * float(patch_success)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


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
# Base scorer
# ---------------------------------------------------------------------------


class ROIScorer:
    """Common initialisation for ROI scoring utilities."""

    def __init__(
        self,
        tracker: ROITracker | None = None,
        calculator: ROICalculator | None = None,
        calculator_factory: Callable[[], ROICalculator] | None = None,
        profile_type: str | None = None,
    ) -> None:
        self.tracker = tracker or ROITracker()
        if calculator is not None:
            self.calculator = calculator
        else:
            try:
                self.calculator = (
                    calculator_factory()
                    if calculator_factory is not None
                    else ROICalculator()
                )
            except Exception as exc:  # pragma: no cover - configuration errors
                raise RuntimeError(
                    "ROICalculator initialization failed. Ensure ROI profile"
                    " configuration is available or provide a calculator"
                ) from exc
        self.profile_type = profile_type or next(iter(self.calculator.profiles))


__all__ = [
    "ROIScorer",
    "Scorecard",
    "EvaluationResult",
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
]
