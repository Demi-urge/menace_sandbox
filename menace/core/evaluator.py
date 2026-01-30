"""Deterministic ROI evaluation utilities."""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable

from menace.errors.exceptions import EvaluatorError

logger = logging.getLogger(__name__)


def _require_decimal(data: Dict[str, Any], field_name: str) -> Decimal:
    if field_name not in data or data[field_name] is None:
        raise EvaluatorError(
            f"Missing required field '{field_name}'",
            details={"field": field_name},
        )

    value = data[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise EvaluatorError(
            f"Field '{field_name}' must be numeric",
            details={"field": field_name, "value_type": type(value).__name__},
        )

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise EvaluatorError(
            f"Field '{field_name}' could not be parsed as a number",
            details={"field": field_name, "value": value},
        ) from exc


def _inputs_used(required_fields: Iterable[str]) -> Dict[str, Any]:
    return {
        "inputs_used": list(required_fields),
    }


def evaluate_roi(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate return on investment (ROI) using deterministic numeric logic.

    ROI is computed deterministically using the formula ``(revenue - cost) / cost``.

    Args:
        input_data: Mapping containing required numeric fields:
            - cost: Total cost for the investment. Must be a positive number.
            - revenue: Total revenue for the investment. Must be numeric.

    Returns:
        Structured evaluation result with the following shape:
            {
                "status": "ok",
                "data": {
                    "roi": Decimal,
                    "components": {
                        "revenue": Decimal,
                        "cost": Decimal,
                        "net_revenue": Decimal,
                    },
                },
                "errors": [],
                "meta": {
                    "inputs_used": ["cost", "revenue"],
                    "formula": "(revenue - cost) / cost",
                },
            }

    Raises:
        EvaluatorError: Raised when required inputs are missing, invalid, or
            non-numeric, or when cost is zero/negative and ROI cannot be
            computed deterministically.
    """
    if not isinstance(input_data, dict):
        raise EvaluatorError(
            "ROI evaluator expected a dict payload",
            details={"received_type": type(input_data).__name__},
        )

    required_fields = ("cost", "revenue")
    cost = _require_decimal(input_data, "cost")
    revenue = _require_decimal(input_data, "revenue")

    if cost <= 0:
        raise EvaluatorError(
            "Cost must be a positive number to compute ROI",
            details={"cost": str(cost)},
        )

    net_revenue = revenue - cost
    roi = net_revenue / cost

    logger.info(
        "Computed ROI",
        extra={
            "roi": str(roi),
            "revenue": str(revenue),
            "cost": str(cost),
        },
    )

    return {
        "status": "ok",
        "data": {
            "roi": roi,
            "components": {
                "revenue": revenue,
                "cost": cost,
                "net_revenue": net_revenue,
            },
        },
        "errors": [],
        "meta": {
            **_inputs_used(required_fields),
            "formula": "(revenue - cost) / cost",
        },
    }
