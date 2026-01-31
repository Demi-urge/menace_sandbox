"""Deterministic ROI delta calculation with explicit validation contracts.

This module performs audit-friendly, linear-time delta computation with no
normalization, weighting, or inference. Metric values must be finite ``int``,
``float``, or ``Decimal`` values; booleans are explicitly disallowed even though
they subclass ``int``. Empty inputs are treated as a valid no-op. Validation
failures return structured error records instead of silently coercing values.
All values are deterministically converted to ``Decimal`` using
``Decimal(str(value))`` for floats and ``Decimal(value)`` for integers to avoid
binary float artifacts.

Canonical response contract for :func:`compute_roi_delta`:

```
{
  "status": "ok" | "error",
  "data": {
    "deltas": {<metric_key>: <delta_number>},
    "total": <sum_of_deltas>
  },
  "errors": [
    {
      "type": "RoiDeltaValidationError",
      "code": "invalid_schema" | "metric_type_error" | "metric_value_error",
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
}
```

Empty inputs return ``status: "ok"`` with ``data.deltas`` as ``{}`` and
``data.total`` set to ``Decimal("0")``.
"""
from __future__ import annotations

import math
from decimal import Decimal
from numbers import Number
from typing import Mapping

from menace_sandbox.stabilization.errors import RoiDeltaValidationError


def _is_valid_metric_value(value: Number | Decimal) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_finite_value(value: Number | Decimal) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    return isinstance(value, (int, float)) and math.isfinite(value)


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
) -> RoiDeltaValidationError:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return RoiDeltaValidationError(
            "Metric value must be an int, float, or Decimal (bool not allowed).",
            code="metric_type_error",
            details={
                "key": key,
                "source": source,
                "value_repr": repr(value),
            },
        )
    return RoiDeltaValidationError(
        "Metric value must be a finite int, float, or Decimal (bool not allowed).",
        code="metric_value_error",
        details={
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
                code="invalid_schema",
                details={"field": "before_metrics"},
            )
        )
    if not isinstance(after_metrics, Mapping):
        errors.append(
            RoiDeltaValidationError(
                "after_metrics must be a mapping.",
                code="invalid_schema",
                details={"field": "after_metrics"},
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
            RoiDeltaValidationError(
                "before_metrics and after_metrics must have identical keys.",
                code="invalid_schema",
                details={"missing": missing_keys, "extra": extra_keys},
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


def compute_roi_delta(
    before_metrics: Mapping[str, Number | Decimal],
    after_metrics: Mapping[str, Number | Decimal],
) -> dict[str, object]:
    """Compute deterministic ROI deltas with strict schema validation.

    Returns a structured payload with status, data, errors, and deterministic
    metadata derived from the metric keys. Empty mappings are treated as a
    valid no-op. Booleans are explicitly rejected as metric values. All
    arithmetic is performed with ``Decimal`` values derived via
    ``Decimal(str(value))`` for floats and ``Decimal(value)`` for integers.
    """

    data = {"deltas": {}, "total": Decimal("0")}

    def _error_payload(
        keys: list[str],
        errors: list[RoiDeltaValidationError],
    ) -> dict[str, object]:
        return {
            "status": "error",
            "data": {"deltas": {}, "total": Decimal("0")},
            "errors": [error.to_record() for error in errors],
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
                    RoiDeltaValidationError(
                        "Computed delta value must be finite.",
                        code="delta_value_error",
                        details={"key": key, "delta_repr": repr(delta_value)},
                    )
                ],
            )
        deltas[key] = delta_value
    total_delta = sum(deltas.values(), Decimal("0"))
    if not _is_finite_value(total_delta):
        return _error_payload(
            keys,
            [
                RoiDeltaValidationError(
                    "Computed total delta must be finite.",
                    code="total_value_error",
                    details={"total_delta_repr": repr(total_delta)},
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


__all__ = [
    "compute_roi_delta",
]
