"""Top-level ROI delta interface with strict schema and validation contract."""
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

    This is a thin, stable wrapper around
    :func:`menace_sandbox.stabilization.roi.compute_roi_delta` that preserves the
    canonical response contract for ROI delta calculations.

    Args:
        before_metrics: Mapping of metric keys to finite numeric values that
            represent the pre-change ROI metrics.
        after_metrics: Mapping of metric keys to finite numeric values that
            represent the post-change ROI metrics.

    Returns:
        A dictionary with the strict ROI delta contract:

        ``{
            "status": "ok" | "error",
            "data": {
                "deltas": {<metric_key>: <delta_decimal>},
                "total": <sum_of_deltas>
            },
            "errors": [
                {
                    "type": "RoiDeltaValidationError",
                    "code": "invalid_schema" | "metric_type_error" | "metric_value_error"
                            | "delta_value_error" | "total_value_error",
                    "message": <string>,
                    "details": <dict>
                }
            ],
            "meta": {
                "keys": [<metric_key>],
                "count": <int>,
                "error_count": <int>,
                "before_count": <int>,
                "after_count": <int>
            }
        }``

        Empty mappings return ``status: "ok"`` with an empty ``deltas`` mapping
        and ``total`` set to ``Decimal("0")``. Validation failures return
        ``status: "error"`` with structured error records and deterministic
        metadata derived from the input keys.
    """

    return _compute_roi_delta(before_metrics, after_metrics)


__all__ = [
    "compute_roi_delta",
]
