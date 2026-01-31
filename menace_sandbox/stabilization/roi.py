"""Deterministic ROI delta calculation with explicit validation contracts.

This module performs audit-friendly, linear-time delta computation with no
normalization, weighting, or inference. Metric values must be finite ``int``,
``float``, or ``Decimal`` values; booleans are explicitly disallowed even though
they subclass ``int``. Empty inputs are treated as a valid no-op. Validation
failures return structured error records instead of silently coercing values.
If any metric is provided as a ``Decimal`` then all values are deterministically
converted to ``Decimal`` using ``Decimal(str(value))`` to avoid binary float
artifacts.
"""
from __future__ import annotations

import math
from decimal import Decimal
from numbers import Number
from typing import Mapping

from menace_sandbox.stabilization.errors import RoiDeltaValidationError


def _sorted_keys(keys: set[str]) -> list[str]:
    return sorted(keys)


def _is_valid_metric_value(value: Number | Decimal) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    return isinstance(value, (int, float)) and math.isfinite(value)


def _to_decimal(value: Number | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _metric_value_error(
    key: str,
    value: Number | Decimal,
    source: str,
) -> RoiDeltaValidationError:
    return RoiDeltaValidationError(
        "Metric value must be a finite int, float, or Decimal (bool not allowed).",
        {
            "key": key,
            "source": source,
            "value_repr": repr(value),
        },
    )


def _validate_metrics(
    before_metrics: Mapping[str, Number | Decimal],
    after_metrics: Mapping[str, Number | Decimal],
) -> tuple[list[str], list[RoiDeltaValidationError]]:
    errors: list[RoiDeltaValidationError] = []
    if not isinstance(before_metrics, Mapping):
        errors.append(
            RoiDeltaValidationError(
                "before_metrics must be a mapping.",
                {"field": "before_metrics"},
            )
        )
    if not isinstance(after_metrics, Mapping):
        errors.append(
            RoiDeltaValidationError(
                "after_metrics must be a mapping.",
                {"field": "after_metrics"},
            )
        )
    if errors:
        return [], errors

    before_keys = set(before_metrics.keys())
    after_keys = set(after_metrics.keys())
    missing_keys = _sorted_keys(before_keys - after_keys)
    extra_keys = _sorted_keys(after_keys - before_keys)
    if missing_keys or extra_keys:
        errors.append(
            RoiDeltaValidationError(
                "before_metrics and after_metrics must have identical keys.",
                {"missing": missing_keys, "extra": extra_keys},
            )
        )
        return _sorted_keys(before_keys | after_keys), errors

    ordered_keys = _sorted_keys(before_keys)
    for key in ordered_keys:
        before_value = before_metrics[key]
        if not _is_valid_metric_value(before_value):
            errors.append(_metric_value_error(key, before_value, "before_metrics"))

        after_value = after_metrics[key]
        if not _is_valid_metric_value(after_value):
            errors.append(_metric_value_error(key, after_value, "after_metrics"))

    return ordered_keys, errors


def compute_roi_delta(
    before_metrics: Mapping[str, Number | Decimal],
    after_metrics: Mapping[str, Number | Decimal],
) -> dict[str, object]:
    """Compute deterministic ROI deltas with strict schema validation.

    Returns a structured payload with status, data, errors, and deterministic
    metadata derived from the metric keys. Empty mappings are treated as a
    valid no-op. Booleans are explicitly rejected as metric values. If any
    metric value is a ``Decimal``, all arithmetic is performed with ``Decimal``
    values derived via ``Decimal(str(value))`` for deterministic precision.
    """

    keys, errors = _validate_metrics(before_metrics, after_metrics)
    if errors:
        return {
            "status": "fail",
            "data": {},
            "errors": [error.to_record() for error in errors],
            "meta": {
                "keys": keys,
                "count": len(keys),
                "error_count": len(errors),
                "before_count": len(before_metrics) if isinstance(before_metrics, Mapping) else 0,
                "after_count": len(after_metrics) if isinstance(after_metrics, Mapping) else 0,
            },
        }

    if not keys:
        return {
            "status": "ok",
            "data": {},
            "errors": [],
            "meta": {
                "keys": [],
                "count": 0,
                "error_count": 0,
                "before_count": 0,
                "after_count": 0,
            },
        }

    use_decimal = any(
        isinstance(before_metrics[key], Decimal) or isinstance(after_metrics[key], Decimal)
        for key in keys
    )
    deltas: dict[str, Number | Decimal] = {}
    for key in keys:
        before_value = before_metrics[key]
        after_value = after_metrics[key]
        if use_decimal:
            before_value = _to_decimal(before_value)
            after_value = _to_decimal(after_value)
        deltas[key] = after_value - before_value
    total_delta = sum(deltas.values(), Decimal("0") if use_decimal else 0)

    return {
        "status": "ok",
        "data": {
            "deltas": deltas,
            "total_delta": total_delta,
        },
        "errors": [],
        "meta": {
            "keys": keys,
            "count": len(keys),
            "error_count": 0,
            "before_count": len(before_metrics),
            "after_count": len(after_metrics),
        },
    }


__all__ = [
    "compute_roi_delta",
]
