from __future__ import annotations

"""Shared utilities for workflow scoring.

This module centralises metric calculations, result containers and the base
ROI scoring class used by both the lightweight :mod:`composite_workflow_scorer`
and the heavier :mod:`roi_scorer` module.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping
from itertools import combinations
import os

_LIGHT_IMPORTS = os.getenv("MENACE_LIGHT_IMPORTS") not in {"", "0", "false", "False", None}

if _LIGHT_IMPORTS:
    class _NumpyStub:
        @staticmethod
        def std(values: Iterable[float]) -> float:
            vals = list(values)
            if len(vals) < 2:
                return 0.0
            mean = sum(vals) / len(vals)
            return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

        @staticmethod
        def corrcoef(a: Iterable[float], b: Iterable[float]) -> list[list[float]]:
            return [[1.0, 0.0], [0.0, 1.0]]

        @staticmethod
        def arange(n: int) -> list[int]:
            return list(range(n))

        @staticmethod
        def polyfit(_x: Iterable[float], _y: Iterable[float], degree: int) -> list[float]:
            return [0.0 for _ in range(degree + 1)]

        @staticmethod
        def average(values: Iterable[float], weights: Iterable[float] | None = None) -> float:
            vals = list(values)
            if not vals:
                return 0.0
            if weights is None:
                return sum(vals) / len(vals)
            wts = list(weights)
            total = sum(wts)
            if not total:
                return sum(vals) / len(vals)
            return sum(v * w for v, w in zip(vals, wts)) / total

    np = _NumpyStub()
else:
    import numpy as np

if _LIGHT_IMPORTS:
    class ROITracker:  # type: ignore[override]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass
    if __package__:
        from .roi_calculator import ROICalculator
    else:  # pragma: no cover - allow execution as script
        from roi_calculator import ROICalculator  # type: ignore
elif __package__:
    from .roi_tracker import ROITracker
    from .roi_calculator import ROICalculator
else:  # pragma: no cover - allow execution as script
    from roi_tracker import ROITracker  # type: ignore
    from roi_calculator import ROICalculator  # type: ignore


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_workflow_synergy(tracker: ROITracker, window: int = 5) -> float:
    """Correlation-based workflow cohesion score.

    Pairwise correlations between module ROI deltas are computed for the most
    recent ``window`` observations.  Each correlation is weighted by the recent
    slope-to-volatility ratio of its historical correlation values to favour
    stable, improving interactions.  The weighted average of pairwise
    correlations is returned.  Computed correlations are cached on ``tracker``
    via :meth:`ROITracker.cache_correlations`.
    """

    module_history = getattr(tracker, "module_deltas", {})
    if not module_history or len(module_history) < 2:
        return 0.0

    correlations: Dict[tuple[str, str], float] = {}
    corrs: list[float] = []
    weights: list[float] = []
    for a, b in combinations(module_history.keys(), 2):
        a_hist = list(module_history.get(a, []))[-window:]
        b_hist = list(module_history.get(b, []))[-window:]
        if len(a_hist) < 2 or len(b_hist) < 2:
            continue
        if np.std(a_hist) == 0 or np.std(b_hist) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a_hist, b_hist)[0, 1])
        pair = tuple(sorted((a, b)))
        correlations[pair] = corr
        history = list(tracker.correlation_history.get(pair, []))[-window:]
        if len(history) >= 2:
            x = np.arange(len(history))
            slope = float(np.polyfit(x, history, 1)[0])
            volatility = float(np.std(history))
            weight = abs(slope) / (volatility + 1e-9)
        else:
            weight = 1.0
        corrs.append(corr)
        weights.append(weight)

    tracker.cache_correlations(correlations)
    return float(np.average(corrs, weights=weights)) if corrs else 0.0


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
