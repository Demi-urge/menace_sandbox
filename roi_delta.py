"""Expose the stabilization ROI delta computation with strict validation."""
from __future__ import annotations

from decimal import Decimal
from numbers import Number
from typing import Mapping

from menace_sandbox.stabilization.roi import compute_roi_delta as _compute_roi_delta


def compute_roi_delta(
    before_metrics: Mapping[str, Number | Decimal],
    after_metrics: Mapping[str, Number | Decimal],
) -> dict[str, object]:
    """Compute deterministic ROI deltas with strict schema validation.

    This delegates to :mod:`menace_sandbox.stabilization.roi` to ensure
    consistent validation behavior and structured error records.
    """
    return _compute_roi_delta(before_metrics, after_metrics)


__all__ = [
    "compute_roi_delta",
]
