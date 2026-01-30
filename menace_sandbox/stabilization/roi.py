from __future__ import annotations

from typing import Any, Mapping


def _coerce_roi(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("roi_score", "roi", "score", "value"):
            if key in value and value[key] is not None:
                return _coerce_roi(value[key])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_roi_delta(prior_roi: Any, current_roi: Any) -> float:
    """Compute ROI delta deterministically as ``current - prior``.

    Missing or non-numeric values are treated as ``0.0``.
    """

    prior_value = _coerce_roi(prior_roi)
    current_value = _coerce_roi(current_roi)
    if prior_value is None:
        prior_value = 0.0
    if current_value is None:
        current_value = 0.0
    return current_value - prior_value


__all__ = ["compute_roi_delta"]
