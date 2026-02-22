"""Compute ROI delta with strict validation and deterministic output."""
from __future__ import annotations

import math
from decimal import Decimal
from numbers import Number
from typing import Mapping

from menace.errors import ValidationError


def compute_roi_delta(
    before_metrics: Mapping[str, Number | Decimal],
    after_metrics: Mapping[str, Number | Decimal],
) -> dict[str, object]:
    """Compute deterministic ROI deltas with strict schema validation."""

    data = {"deltas": {}, "total": Decimal("0")}

    def _error_payload(
        keys: list[str],
        errors: list[ValidationError],
    ) -> dict[str, object]:
        return {
            "status": "error",
            "data": {"deltas": {}, "total": Decimal("0")},
            "errors": [_error_record(error) for error in errors],
            "meta": {
                "keys": keys,
                "count": len(keys),
                "error_count": len(errors),
                "before_count": len(before_metrics) if isinstance(before_metrics, Mapping) else 0,
                "after_count": len(after_metrics) if isinstance(after_metrics, Mapping) else 0,
            },
        }

    keys, errors = _validate_metrics(before_metrics, after_metrics)
    if errors:
        return _error_payload(keys, errors)

    if not keys:
        return {
            "status": "ok",
            "data": data,
            "errors": [],
            "meta": {
                "keys": [],
                "count": 0,
                "error_count": 0,
                "before_count": 0,
                "after_count": 0,
            },
        }

    deltas: dict[str, Decimal] = {}
    for key in keys:
        before_value = _to_decimal(before_metrics[key])
        after_value = _to_decimal(after_metrics[key])
        delta_value = after_value - before_value
        if not _is_finite_value(delta_value):
            return _error_payload(
                keys,
                [
                    ValidationError(
                        "Computed delta value must be finite.",
                        details={
                            "code": "delta_value_error",
                            "key": key,
                            "delta_repr": repr(delta_value),
                        },
                    )
                ],
            )
        deltas[key] = delta_value

    total_delta = sum(deltas.values(), Decimal("0"))
    if not _is_finite_value(total_delta):
        return _error_payload(
            keys,
            [
                ValidationError(
                    "Computed total delta must be finite.",
                    details={
                        "code": "total_value_error",
                        "total_delta_repr": repr(total_delta),
                    },
                )
            ],
        )

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
            "before_count": len(before_metrics),
            "after_count": len(after_metrics),
        },
    }


def _error_record(error: ValidationError) -> dict[str, object]:
    record = error.to_dict()
    code = error.details.get("code")
    if code is not None:
        record["code"] = code
    return record


def _is_valid_metric_value(value: Number | Decimal) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, Number):
        try:
            return math.isfinite(value)
        except TypeError:
            return False
    return False


def _is_finite_value(value: Number | Decimal) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, Number):
        try:
            return math.isfinite(value)
        except TypeError:
            return False
    return False


def _to_decimal(value: Number | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    return Decimal(str(value))


def _metric_value_error(
    key: str,
    value: Number | Decimal,
    source: str,
) -> ValidationError:
    if isinstance(value, bool) or not isinstance(value, Number):
        return ValidationError(
            "Metric value must be a finite number (bool not allowed).",
            details={
                "code": "metric_type_error",
                "key": key,
                "source": source,
                "value_repr": repr(value),
            },
        )
    return ValidationError(
        "Metric value must be finite (bool not allowed).",
        details={
            "code": "metric_value_error",
            "key": key,
            "source": source,
            "value_repr": repr(value),
        },
    )


def _validate_metrics(
    before_metrics: Mapping[str, Number | Decimal],
    after_metrics: Mapping[str, Number | Decimal],
) -> tuple[list[str], list[ValidationError]]:
    errors: list[ValidationError] = []
    if not isinstance(before_metrics, Mapping):
        errors.append(
            ValidationError(
                "before_metrics must be a mapping.",
                details={"code": "invalid_schema", "field": "before_metrics"},
            )
        )
    if not isinstance(after_metrics, Mapping):
        errors.append(
            ValidationError(
                "after_metrics must be a mapping.",
                details={"code": "invalid_schema", "field": "after_metrics"},
            )
        )
    if errors:
        return [], errors

    before_keys = set(before_metrics.keys())
    after_keys = set(after_metrics.keys())
    missing_keys = [key for key in before_metrics.keys() if key not in after_keys]
    extra_keys = [key for key in after_metrics.keys() if key not in before_keys]
    if missing_keys or extra_keys:
        errors.append(
            ValidationError(
                "before_metrics and after_metrics must have identical keys.",
                details={
                    "code": "invalid_schema",
                    "missing": missing_keys,
                    "extra": extra_keys,
                },
            )
        )
        ordered_keys = list(before_metrics.keys()) + extra_keys
        return ordered_keys, errors

    ordered_keys = list(before_metrics.keys())
    for key in ordered_keys:
        before_value = before_metrics[key]
        if not _is_valid_metric_value(before_value):
            errors.append(_metric_value_error(key, before_value, "before_metrics"))

        after_value = after_metrics[key]
        if not _is_valid_metric_value(after_value):
            errors.append(_metric_value_error(key, after_value, "after_metrics"))

    return ordered_keys, errors


__all__ = [
    "compute_roi_delta",
]
