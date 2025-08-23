from __future__ import annotations

"""Append-only logger recording ROI forecasts and threshold details."""

import json
import threading
import time
from pathlib import Path
from typing import Iterable, Mapping, Any


class ForecastLogger:
    """Write forecast information to a JSON lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, data: Mapping[str, Any]) -> None:
        record = {"timestamp": int(time.time())}
        record.update({k: v for k, v in data.items()})
        line = json.dumps(record)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


def log_forecast_record(
    logger: "ForecastLogger" | None,
    workflow_id: str,
    projections: Iterable[Mapping[str, Any]],
    confidence: float | None,
    reason_codes: Iterable[str],
    upgrade_id: str | None = None,
) -> None:
    """Helper to persist a structured foresight promotion record.

    Parameters
    ----------
    logger:
        Instance of :class:`ForecastLogger` or ``None``.
    workflow_id:
        Identifier of the workflow being evaluated.
    projections:
        Iterable of projection mappings, typically produced by the
        forecaster.
    confidence:
        Confidence score associated with the forecast.
    reason_codes:
        Iterable of reason codes explaining promotion downgrade decisions.
    upgrade_id:
        Optional identifier of the forecast run.
    """

    if logger is None:
        return
    payload: dict[str, Any] = {
        "workflow_id": workflow_id,
        "projections": list(projections),
        "confidence": confidence,
        "reason_codes": list(reason_codes),
    }
    if upgrade_id is not None:
        payload["upgrade_id"] = upgrade_id
    try:
        logger.log(payload)
    except Exception:
        pass


__all__ = ["ForecastLogger", "log_forecast_record"]
