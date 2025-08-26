from __future__ import annotations

"""Shared workflow metric helpers."""

from typing import Any, Iterable, Mapping

import numpy as np


def compute_workflow_synergy(
    roi_history: Iterable[float],
    module_history: Mapping[str, Iterable[float]],
    window: int = 5,
) -> float:
    """Mean correlation between module ROI deltas and overall ROI.

    Examines the ``window`` most recent ROI values along with the corresponding
    per-module ROI deltas. For each module a Pearson correlation coefficient with
    the overall ROI history is computed and the average of all valid correlations
    is returned. Modules with insufficient variance or history are ignored.
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
        correlations.append(float(np.corrcoef(roi_list, mod_list)[0, 1]))
    return float(np.mean(correlations)) if correlations else 0.0


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


__all__ = [
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
]
