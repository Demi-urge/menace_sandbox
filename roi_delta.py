"""Compute ROI deltas with strict validation and deterministic metadata."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

from menace.errors import ValidationError


def compute_roi_delta(
    before_metrics: Mapping[str, float],
    after_metrics: Mapping[str, float],
) -> dict[str, object]:
    """Compute deterministic ROI deltas with strict schema validation.

    Args:
        before_metrics: Mapping of metric keys to finite numeric values that
            represent the pre-change ROI metrics.
        after_metrics: Mapping of metric keys to finite numeric values that
            represent the post-change ROI metrics.

    Returns:
        A dictionary with the ROI delta contract.
    """
    if not isinstance(before_metrics, Mapping) or not isinstance(after_metrics, Mapping):
        error = ValidationError(
            "before_metrics and after_metrics must be mappings.",
            {
                "before_is_mapping": isinstance(before_metrics, Mapping),
                "after_is_mapping": isinstance(after_metrics, Mapping),
            },
        )
        return _failed_response(error, _build_meta({}, {}))

    before_keys = set(before_metrics.keys())
    after_keys = set(after_metrics.keys())
    if before_keys != after_keys:
        error = ValidationError(
            "before_metrics and after_metrics must have identical keys.",
            {
                "missing": sorted(before_keys - after_keys),
                "extra": sorted(after_keys - before_keys),
            },
        )
        return _failed_response(error, _build_meta(before_metrics, after_metrics))

    deltas: dict[str, float] = {}
    for key in sorted(before_keys):
        before_value = before_metrics[key]
        after_value = after_metrics[key]
        error = _validate_metric_value(key, before_value, "before_metrics")
        if error:
            return _failed_response(error, _build_meta(before_metrics, after_metrics))
        error = _validate_metric_value(key, after_value, "after_metrics")
        if error:
            return _failed_response(error, _build_meta(before_metrics, after_metrics))
        deltas[key] = float(after_value) - float(before_value)

    total_delta = sum(deltas.values(), 0.0)
    return {
        "status": "ok",
        "data": {"deltas": deltas, "total_delta": total_delta},
        "errors": [],
        "meta": {
            "metric_count": len(deltas),
            "input_hash": _hash_inputs(before_metrics, after_metrics),
        },
    }


def _validate_metric_value(key: str, value: float, source: str) -> ValidationError | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ValidationError(
            "Metric values must be numeric and finite.",
            {"key": key, "source": source, "value_repr": repr(value)},
        )
    if not math.isfinite(value):
        return ValidationError(
            "Metric values must be numeric and finite.",
            {"key": key, "source": source, "value_repr": repr(value)},
        )
    return None


def _hash_inputs(before_metrics: Mapping[str, float], after_metrics: Mapping[str, float]) -> str:
    payload = {
        "after": sorted(_format_items(after_metrics)),
        "before": sorted(_format_items(before_metrics)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _format_items(metrics: Mapping[str, float]) -> list[tuple[str, str]]:
    return [(key, repr(value)) for key, value in metrics.items()]


def _build_meta(before_metrics: Mapping[str, float], after_metrics: Mapping[str, float]) -> dict[str, object]:
    if not isinstance(before_metrics, Mapping) or not isinstance(after_metrics, Mapping):
        return {"metric_count": 0, "input_hash": None}
    return {
        "metric_count": len(set(before_metrics.keys()) | set(after_metrics.keys())),
        "input_hash": _hash_inputs(before_metrics, after_metrics),
    }


def _failed_response(error: ValidationError, meta: dict[str, object]) -> dict[str, object]:
    return {
        "status": "failed",
        "data": {},
        "errors": [error.to_dict()],
        "meta": meta,
    }


__all__ = [
    "compute_roi_delta",
]
