"""Deterministic ROI evaluation utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict

from menace.errors.exceptions import EvaluationError

logger = logging.getLogger(__name__)


def _require_numeric(data: Dict[str, Any], field_name: str) -> float:
    if field_name not in data or data[field_name] is None:
        raise EvaluationError(
            f"Missing required field '{field_name}'",
            details={"field": field_name},
        )

    value = data[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(
            f"Field '{field_name}' must be numeric",
            details={"field": field_name, "value_type": type(value).__name__},
        )
    return float(value)


def evaluate_roi(data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate deterministic ROI from provided fields.

    The ROI formula is deterministic and uses only the provided numeric inputs:

    ``roi = ((benefit - cost) / cost) * risk_adjustment``

    Invariants:
    - ``data`` must be a mapping containing numeric ``cost``, ``benefit``,
      and ``risk_adjustment`` values.
    - ``cost`` must be non-zero to avoid division by zero.
    - Missing, ``None``, or non-numeric values raise ``EvaluationError``.

    Args:
        data: Input payload containing required numeric fields:
            - benefit: Total benefit for the workflow.
            - cost: Total cost for the workflow.
            - risk_adjustment: Multiplicative risk adjustment factor.

    Returns:
        Structured result containing the numeric ROI, a component breakdown,
        and metadata describing the inputs used.

    Raises:
        EvaluationError: If input payload is invalid or ROI cannot be
            calculated deterministically.
    """
    if not isinstance(data, dict):
        raise EvaluationError(
            "ROI evaluator expected a dict payload",
            details={"received_type": type(data).__name__},
        )

    benefit = _require_numeric(data, "benefit")
    cost = _require_numeric(data, "cost")
    risk_adjustment = _require_numeric(data, "risk_adjustment")

    if cost == 0:
        raise EvaluationError("Cost must be non-zero to compute ROI", details={"cost": cost})

    net_benefit = benefit - cost
    base_roi = net_benefit / cost
    roi = base_roi * risk_adjustment

    logger.info(
        "Computed ROI",
        extra={
            "roi": roi,
            "benefit": benefit,
            "cost": cost,
            "risk_adjustment": risk_adjustment,
        },
    )

    return {
        "roi": roi,
        "components": {
            "benefit": benefit,
            "cost": cost,
            "risk_adjustment": risk_adjustment,
            "net_benefit": net_benefit,
            "base_roi": base_roi,
        },
        "metadata": {
            "inputs_used": ["benefit", "cost", "risk_adjustment"],
        },
    }
