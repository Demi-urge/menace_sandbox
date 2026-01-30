"""Deterministic ROI evaluation utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict

from menace.errors.exceptions import CalculationError, MalformedInputError, MissingFieldError

logger = logging.getLogger(__name__)


def _require_numeric(data: Dict[str, Any], field_name: str) -> float:
    if field_name not in data or data[field_name] is None:
        raise MissingFieldError(field_name)

    value = data[field_name]
    if not isinstance(value, (int, float)):
        raise MalformedInputError(
            f"Field '{field_name}' must be numeric",
            details={"field": field_name, "value_type": type(value).__name__},
        )
    return float(value)


def evaluate_roi(data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate deterministic ROI from provided fields.

    Args:
        data: Input payload containing required numeric fields:
            - revenue: Total revenue for the workflow.
            - cost: Total cost for the workflow.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta.

    Raises:
        MissingFieldError: If required fields are missing or None.
        MalformedInputError: If input payload is not a dict or fields are malformed.
        CalculationError: If ROI cannot be calculated deterministically.
    """
    if not isinstance(data, dict):
        raise MalformedInputError(
            "ROI evaluator expected a dict payload",
            details={"received_type": type(data).__name__},
        )

    revenue = _require_numeric(data, "revenue")
    cost = _require_numeric(data, "cost")

    if cost == 0:
        raise CalculationError("Cost must be non-zero to compute ROI", details={"cost": cost})

    roi = (revenue - cost) / cost

    logger.info(
        "Computed ROI",
        extra={"roi": roi, "revenue": revenue, "cost": cost},
    )

    return {
        "status": "ok",
        "data": {
            "roi": roi,
            "revenue": revenue,
            "cost": cost,
        },
        "errors": [],
        "meta": {
            "inputs": {
                "revenue": revenue,
                "cost": cost,
            }
        },
    }
