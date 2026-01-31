"""Deterministic ROI delta calculation with strict schema validation."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import math
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RoiDeltaError(Exception):
    """Base error for ROI delta validation."""

    code: str
    message: str
    key: str | None = None
    value: Any = None

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "key": self.key,
            "value_repr": repr(self.value),
        }


class InputTypeError(RoiDeltaError):
    """Raised when inputs are not mappings."""

    def __init__(self, *, key: str | None, value: Any) -> None:
        super().__init__(
            code="input_type_error",
            message="Expected a mapping of ROI metrics.",
            key=key,
            value=value,
        )


class SchemaMismatchError(RoiDeltaError):
    """Raised when before/after schemas differ."""

    def __init__(self, *, missing_keys: list[str], extra_keys: list[str]) -> None:
        super().__init__(
            code="schema_mismatch",
            message="Before/after metric keys do not match.",
            key=None,
            value={"missing_keys": missing_keys, "extra_keys": extra_keys},
        )


class NonNumericValueError(RoiDeltaError):
    """Raised when a value is not numeric."""

    def __init__(self, *, key: str, value: Any) -> None:
        super().__init__(
            code="non_numeric_value",
            message="Metric value is not numeric.",
            key=key,
            value=value,
        )


class NonFiniteValueError(RoiDeltaError):
    """Raised when a numeric value is NaN or infinite."""

    def __init__(self, *, key: str, value: Any) -> None:
        super().__init__(
            code="non_finite_value",
            message="Metric value is not finite.",
            key=key,
            value=value,
        )


def _schema_hash(keys: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for key in keys:
        hasher.update(key.encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


def _coerce_numeric(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise NonNumericValueError(key=key, value=value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise NonFiniteValueError(key=key, value=value)
    try:
        coerced = float(value)
    except (OverflowError, ValueError):
        raise NonFiniteValueError(key=key, value=value) from None
    if not math.isfinite(coerced):
        raise NonFiniteValueError(key=key, value=value)
    return coerced


def _validate_mappings(
    before_metrics: Mapping[str, Any] | None,
    after_metrics: Mapping[str, Any] | None,
) -> tuple[list[str], list[RoiDeltaError]]:
    errors: list[RoiDeltaError] = []
    if before_metrics is None or not isinstance(before_metrics, Mapping):
        errors.append(InputTypeError(key="before_metrics", value=before_metrics))
    if after_metrics is None or not isinstance(after_metrics, Mapping):
        errors.append(InputTypeError(key="after_metrics", value=after_metrics))
    if errors:
        return [], errors

    before_keys = list(before_metrics.keys())
    missing_keys: list[str] = []
    for key in before_keys:
        if key not in after_metrics:
            missing_keys.append(key)

    extra_keys: list[str] = []
    for key in after_metrics.keys():
        if key not in before_metrics:
            extra_keys.append(key)

    if missing_keys or extra_keys:
        errors.append(
            SchemaMismatchError(missing_keys=missing_keys, extra_keys=extra_keys)
        )
        return before_keys, errors

    return before_keys, errors


def compute_roi_delta(
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute deterministic ROI deltas with strict schema validation.

    Returns a structured payload with status, data, errors, and deterministic
    metadata derived from the metric keys.
    """

    keys, errors = _validate_mappings(before_metrics, after_metrics)
    if errors:
        return {
            "status": "error",
            "data": None,
            "errors": [error.to_record() for error in errors],
            "meta": {
                "key_count": len(keys),
                "keys": keys,
                "schema_hash": _schema_hash(keys),
            },
        }

    deltas: dict[str, float] = {}
    delta_total = 0.0
    value_errors: list[RoiDeltaError] = []
    for key in keys:
        try:
            before_value = _coerce_numeric(before_metrics[key], key=key)
        except RoiDeltaError as exc:
            value_errors.append(exc)
            continue
        try:
            after_value = _coerce_numeric(after_metrics[key], key=key)
        except RoiDeltaError as exc:
            value_errors.append(exc)
            continue
        delta = after_value - before_value
        deltas[key] = delta
        delta_total += delta

    if value_errors:
        return {
            "status": "error",
            "data": None,
            "errors": [error.to_record() for error in value_errors],
            "meta": {
                "key_count": len(keys),
                "keys": keys,
                "schema_hash": _schema_hash(keys),
            },
        }

    return {
        "status": "ok",
        "data": {
            "deltas": deltas,
            "delta_total": delta_total,
        },
        "errors": [],
        "meta": {
            "key_count": len(keys),
            "keys": keys,
            "schema_hash": _schema_hash(keys),
        },
    }


__all__ = [
    "RoiDeltaError",
    "InputTypeError",
    "SchemaMismatchError",
    "NonNumericValueError",
    "NonFiniteValueError",
    "compute_roi_delta",
]
