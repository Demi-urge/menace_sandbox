"""Deterministic ROI delta calculation with strict schema validation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class InvalidMetricsSchemaError(Exception):
    """Raised when the metric inputs are invalid or schema mismatched."""

    message: str
    details: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "code": "invalid_schema",
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class MetricTypeError(Exception):
    """Raised when a metric value has the wrong type."""

    key: str
    value: Any
    message: str = "Metric value must be an int or float (bool not allowed)."

    def to_record(self) -> dict[str, Any]:
        return {
            "code": "metric_type_error",
            "message": self.message,
            "key": self.key,
            "value_repr": repr(self.value),
        }


@dataclass(frozen=True)
class MetricValueError(Exception):
    """Raised when a metric value is not finite."""

    key: str
    value: Any
    message: str = "Metric value must be finite."

    def to_record(self) -> dict[str, Any]:
        return {
            "code": "metric_value_error",
            "message": self.message,
            "key": self.key,
            "value_repr": repr(self.value),
        }


def _format_keys(keys: set[object]) -> list[str]:
    return sorted([str(key) for key in keys])


def _validate_metrics(
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
) -> tuple[list[str], list[Exception]]:
    errors: list[Exception] = []
    if not isinstance(before_metrics, Mapping):
        errors.append(
            InvalidMetricsSchemaError(
                message="before_metrics must be a mapping.",
                details={"input": "before_metrics", "value_repr": repr(before_metrics)},
            )
        )
    if not isinstance(after_metrics, Mapping):
        errors.append(
            InvalidMetricsSchemaError(
                message="after_metrics must be a mapping.",
                details={"input": "after_metrics", "value_repr": repr(after_metrics)},
            )
        )
    if errors:
        return [], errors

    before_keys = set(before_metrics.keys())
    after_keys = set(after_metrics.keys())
    missing_keys = _format_keys(before_keys - after_keys)
    extra_keys = _format_keys(after_keys - before_keys)
    if missing_keys or extra_keys:
        errors.append(
            InvalidMetricsSchemaError(
                message="before_metrics and after_metrics must have identical keys.",
                details={"missing_keys": missing_keys, "extra_keys": extra_keys},
            )
        )
        return _format_keys(before_keys), errors

    invalid_key_types = [
        key for key in before_keys if not isinstance(key, str)
    ]
    if invalid_key_types:
        errors.append(
            InvalidMetricsSchemaError(
                message="All metric keys must be strings.",
                details={"invalid_keys": [repr(key) for key in invalid_key_types]},
            )
        )
        return _format_keys(before_keys), errors

    ordered_keys = sorted(before_keys)
    for key in ordered_keys:
        value = before_metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(MetricTypeError(key=key, value=value))
            continue
        if not math.isfinite(float(value)):
            errors.append(MetricValueError(key=key, value=value))

        value = after_metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(MetricTypeError(key=key, value=value))
            continue
        if not math.isfinite(float(value)):
            errors.append(MetricValueError(key=key, value=value))

    return ordered_keys, errors


def compute_roi_delta(
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute deterministic ROI deltas with strict schema validation.

    Returns a structured payload with status, data, errors, and deterministic
    metadata derived from the metric keys.
    """

    keys, errors = _validate_metrics(before_metrics, after_metrics)
    if errors:
        return {
            "status": "error",
            "data": {},
            "errors": [
                error.to_record()
                for error in errors
                if hasattr(error, "to_record")
            ],
            "meta": {
                "input_keys": keys,
                "key_count": len(keys),
                "error_count": len(errors),
            },
        }

    deltas = {key: float(after_metrics[key]) - float(before_metrics[key]) for key in keys}

    return {
        "status": "ok",
        "data": {
            "delta": deltas,
        },
        "errors": [],
        "meta": {
            "input_keys": keys,
            "key_count": len(keys),
            "error_count": 0,
        },
    }


__all__ = [
    "InvalidMetricsSchemaError",
    "MetricTypeError",
    "MetricValueError",
    "compute_roi_delta",
]
