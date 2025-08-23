from __future__ import annotations

"""Foresight promotion gate evaluating forecast projections."""

from dataclasses import asdict
from typing import Iterable, List, Any, Dict

from upgrade_forecaster import UpgradeForecaster
from forecast_logger import ForecastLogger
try:  # pragma: no cover - optional dependency
    from workflow_graph import WorkflowGraph
except Exception:  # pragma: no cover
    WorkflowGraph = object  # type: ignore[misc,assignment]


_reason_roi = "projected_roi_below_threshold"
_reason_conf = "low_confidence"
_reason_collapse = "roi_collapse_risk"
_reason_dag = "negative_dag_impact"


def _log(logger: ForecastLogger | None, payload: dict) -> None:
    if logger is None:
        return
    try:
        logger.log(payload)
    except Exception:  # pragma: no cover - best effort
        pass


def is_foresight_safe_to_promote(
    workflow_id: str,
    patch: Iterable[str] | str,
    forecaster: UpgradeForecaster,
    workflow_graph: WorkflowGraph | None = None,
    *,
    roi_threshold: float = 0.0,
    confidence_threshold: float = 0.6,
) -> tuple[bool, List[str], Dict[str, Any]]:
    """Assess whether ``patch`` may be promoted based on forecasted metrics.

    Parameters
    ----------
    workflow_id:
        Identifier of the workflow being evaluated.
    patch:
        Collection of diff hunks or a summary string describing the upgrade.
    forecaster:
        Forecasting helper providing :meth:`forecast` and a ``tracker`` with
        :meth:`predict_roi_collapse`.
    workflow_graph:
        Optional graph used to simulate impact waves.
    roi_threshold:
        Minimum acceptable ROI projection for all cycles.
    confidence_threshold:
        Minimum forecast confidence required for promotion.
    
    Returns
    -------
    tuple[bool, List[str], Dict[str, Any]]
        ``safe`` decision flag, list of ``reasons`` for rejection and a
        ``forecast`` mapping containing projection details.
    """

    forecast = forecaster.forecast(workflow_id, patch)
    reasons: List[str] = []

    # Basic projection checks -------------------------------------------------
    if any(p.roi < roi_threshold for p in forecast.projections):
        reasons.append(_reason_roi)

    if forecast.confidence < confidence_threshold:
        reasons.append(_reason_conf)

    # Collapse risk -----------------------------------------------------------
    tracker = getattr(forecaster, "tracker", None)
    if tracker is not None:
        try:
            collapse = tracker.predict_roi_collapse(workflow_id)
            if collapse.get("risk") == "Immediate collapse risk" or collapse.get(
                "brittle"
            ):
                reasons.append(_reason_collapse)
        except Exception:  # pragma: no cover - best effort
            pass

    # Impact wave -------------------------------------------------------------
    if workflow_graph is not None:
        try:
            roi_delta = forecast.projections[0].roi if forecast.projections else 0.0
            synergy_delta = (
                -forecast.projections[0].decay if forecast.projections else 0.0
            )
            impacts = workflow_graph.simulate_impact_wave(
                workflow_id, float(roi_delta), float(synergy_delta)
            )
            for vals in impacts.values():
                if vals.get("roi", 0.0) < 0:
                    reasons.append(_reason_dag)
                    break
        except Exception:  # pragma: no cover - best effort
            pass

    safe = not reasons

    forecast_info: Dict[str, Any] = {
        "projections": [asdict(p) for p in forecast.projections],
        "confidence": forecast.confidence,
        "upgrade_id": forecast.upgrade_id,
    }

    _log(
        getattr(forecaster, "logger", None),
        {
            "workflow_id": workflow_id,
            "forecast": forecast_info,
            "reason_codes": reasons,
            "decision": safe,
        },
    )

    return safe, reasons, forecast_info


__all__ = ["is_foresight_safe_to_promote"]
