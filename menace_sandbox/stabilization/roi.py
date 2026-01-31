"""Deterministic ROI delta calculation with explicit validation contracts.

This module performs audit-friendly, linear-time delta computation with no
normalization, weighting, or inference. Metric values must be numeric and
finite; booleans are explicitly disallowed even though they subclass ``int``.
Empty inputs are treated as a valid no-op and return zero totals. Validation
failures return structured error records instead of silently coercing values.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class InvalidMetricsSchemaError(Exception):
    """Raised when the metric inputs are invalid or schema mismatched."""

    message: str
    details: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "invalid_schema",
            "code": "invalid_schema",
            "message": self.message,
            "field": self.details.get("field", "schema"),
            "details": self.details,
        }


@dataclass(frozen=True)
class MetricTypeError(Exception):
    """Raised when a metric value has the wrong type."""

    key: str
    value: Any
    source: str
    message: str = "Metric value must be numeric (bool not allowed)."

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "metric_type_error",
            "code": "metric_type_error",
            "message": (
                f"{self.message} Offending key: {self.key} (source: {self.source})."
            ),
            "key": self.key,
            "value_repr": repr(self.value),
            "source": self.source,
        }


@dataclass(frozen=True)
class MetricValueError(Exception):
    """Raised when a metric value is not finite."""

    key: str
    value: Any
    source: str
    message: str = "Metric value must be finite."

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "metric_value_error",
            "code": "metric_value_error",
            "message": (
                f"{self.message} Offending key: {self.key} (source: {self.source})."
            ),
            "key": self.key,
            "value_repr": repr(self.value),
            "source": self.source,
        }


def _format_keys(keys: set[object]) -> list[str]:
    return sorted([str(key) for key in keys])


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, Fraction):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _coerce_numeric(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _validate_metrics(
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
) -> tuple[list[str], list[Exception]]:
    errors: list[Exception] = []
    if not isinstance(before_metrics, Mapping):
        errors.append(
            InvalidMetricsSchemaError(
                message="before_metrics must be a mapping.",
                details={"field": "before_metrics"},
            )
        )
        return [], errors
    if not isinstance(after_metrics, Mapping):
        errors.append(
            InvalidMetricsSchemaError(
                message="after_metrics must be a mapping.",
                details={"field": "after_metrics"},
            )
        )
        return [], errors

    before_keys = set(before_metrics.keys())
    after_keys = set(after_metrics.keys())
    missing_keys = _format_keys(before_keys - after_keys)
    extra_keys = _format_keys(after_keys - before_keys)
    if missing_keys or extra_keys:
        errors.append(
            InvalidMetricsSchemaError(
                message=(
                    "before_metrics and after_metrics must have identical keys. "
                    f"Missing: {missing_keys}. Extra: {extra_keys}."
                ),
                details={
                    "field": "keys",
                    "missing": missing_keys,
                    "extra": extra_keys,
                },
            )
        )
        return _format_keys(before_keys | after_keys), errors

    invalid_key_types = [
        key for key in before_keys if not isinstance(key, str)
    ]
    if invalid_key_types:
        errors.append(
            InvalidMetricsSchemaError(
                message="All metric keys must be strings.",
                details={
                    "field": "keys",
                    "invalid_keys": [repr(key) for key in invalid_key_types],
                },
            )
        )
        return _format_keys(before_keys), errors

    ordered_keys = sorted(before_keys)
    for key in ordered_keys:
        before_value = before_metrics[key]
        if isinstance(before_value, bool) or not isinstance(before_value, (Real, Decimal)):
            errors.append(
                MetricTypeError(key=key, value=before_value, source="before_metrics")
            )
        elif not _is_finite_number(before_value):
            errors.append(
                MetricValueError(key=key, value=before_value, source="before_metrics")
            )

        after_value = after_metrics[key]
        if isinstance(after_value, bool) or not isinstance(after_value, (Real, Decimal)):
            errors.append(
                MetricTypeError(key=key, value=after_value, source="after_metrics")
            )
        elif not _is_finite_number(after_value):
            errors.append(
                MetricValueError(key=key, value=after_value, source="after_metrics")
            )

    return ordered_keys, errors


def compute_roi_delta(
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute deterministic ROI deltas with strict schema validation.

    Returns a structured payload with status, data, errors, and deterministic
    metadata derived from the metric keys. Empty mappings are treated as a
    valid no-op. Booleans are explicitly rejected as metric values.
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
                "keys": keys,
                "count": len(keys),
                "error_count": len(errors),
            },
        }

    deltas: dict[str, Decimal] = {}
    for key in keys:
        before_value = _coerce_numeric(before_metrics[key])
        after_value = _coerce_numeric(after_metrics[key])
        deltas[key] = after_value - before_value
    total_delta = sum(deltas.values(), Decimal("0"))

    return {
        "status": "ok",
        "data": {
            "deltas": deltas,
            "total": total_delta,
        },
        "errors": [],
        "meta": {
            "keys": keys,
            "count": len(keys),
            "error_count": 0,
        },
    }


__all__ = [
    "InvalidMetricsSchemaError",
    "MetricTypeError",
    "MetricValueError",
    "compute_roi_delta",
]
