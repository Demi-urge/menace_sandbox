"""Project patch impact over forthcoming improvement cycles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import json
import time
import hashlib
import numpy as np

try:  # pragma: no cover - allow package or local imports
    from .foresight_tracker import ForesightTracker
except Exception:  # pragma: no cover
    from foresight_tracker import ForesightTracker  # type: ignore
try:  # pragma: no cover - optional dependency
    from sandbox_runner.environment import simulate_temporal_trajectory
except Exception:  # pragma: no cover
    def simulate_temporal_trajectory(*args, **kwargs):  # type: ignore[override]
        return None


@dataclass(frozen=True)
class CycleProjection:
    """Per-cycle forecast entry."""

    cycle: float
    roi: float
    risk: float
    confidence: float
    decay: float


@dataclass(frozen=True)
class ForecastResult:
    """Container for forecast projections and overall confidence."""

    projections: List[CycleProjection]
    confidence: float
    upgrade_id: str


class UpgradeForecaster:
    """Forecast ROI and stability for potential upgrades.

    Parameters
    ----------
    tracker:
        Instance providing historic metrics.
    horizon:
        Number of cycles to forecast. Clamped to the range 3–5.
    records_base:
        Directory where JSON forecast records are written.
    logger:
        Optional :class:`ForecastLogger` used for structured logging.
    """

    def __init__(
        self,
        tracker: ForesightTracker,
        horizon: int = 5,
        records_base: str | Path = "forecast_records",
        logger: object | None = None,
    ) -> None:
        self.tracker = tracker
        self.horizon = max(3, min(5, int(horizon)))
        self.records_base = Path(records_base)
        self.records_base.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    # ------------------------------------------------------------------
    def forecast(
        self,
        workflow_id: str,
        patch: Iterable[str] | str,
        cycles: int | None = None,
    ) -> ForecastResult:
        """Return cycle projections and overall forecast confidence."""

        cycles = self.horizon if cycles is None else max(3, min(5, int(cycles)))
        wf_id = str(workflow_id)

        patch_repr = list(patch) if not isinstance(patch, str) else patch
        try:
            patch_serial = json.dumps(patch_repr, sort_keys=True)
        except Exception:
            patch_serial = repr(patch_repr)
        patch_hash = hashlib.sha256(patch_serial.encode("utf8")).hexdigest()
        upgrade_id = patch_hash

        # Simulate the patched workflow to obtain prospective metrics
        try:
            roi_tracker = simulate_temporal_trajectory(
                wf_id, patch, foresight_tracker=self.tracker
            )
        except Exception as exc:  # pragma: no cover - exercised in tests
            roi_tracker = None
            roi_hist: List[float] = []
            conf_hist: List[float] = []
            entropy_hist: List[float] = []
            risk_hist: List[float] = []
            samples = 0
            cold_start = True
            if self.logger is not None:
                try:
                    self.logger.log(
                        {
                            "workflow_id": wf_id,
                            "patch": patch_repr,
                            "error": repr(exc),
                            "event": "simulate_temporal_trajectory_failed",
                        }
                    )
                except Exception:
                    pass
        else:
            roi_hist = getattr(roi_tracker, "roi_history", []) or []
            conf_hist = getattr(roi_tracker, "confidence_history", []) or []
            metrics_hist = getattr(roi_tracker, "metrics_history", {}) or {}
            entropy_hist = metrics_hist.get("synergy_shannon_entropy", []) or []
            risk_hist = metrics_hist.get("synergy_risk_index", []) or []
            samples = len(self.tracker.history.get(wf_id, []))
            cold_start = self.tracker.is_cold_start(wf_id)

        projections: List[CycleProjection] = []

        if cold_start:
            # Blend template curves with simulated metrics
            template_roi = self.tracker.get_template_curve(wf_id)
            get_entropy_template = getattr(
                self.tracker, "get_entropy_template_curve", None
            )
            template_entropy: List[float] = []
            if callable(get_entropy_template):
                try:
                    template_entropy = list(get_entropy_template(wf_id) or [])
                except Exception:
                    template_entropy = []
            get_risk_template = getattr(
                self.tracker, "get_risk_template_curve", None
            )
            template_risk: List[float] = []
            if callable(get_risk_template):
                try:
                    template_risk = list(get_risk_template(wf_id) or [])
                except Exception:
                    template_risk = []

            alpha = min(1.0, samples / 5.0)
            for i in range(1, cycles + 1):
                sim_roi = roi_hist[i - 1] if i - 1 < len(roi_hist) else (
                    roi_hist[-1] if roi_hist else 0.0
                )
                templ_roi = (
                    template_roi[i - 1]
                    if i - 1 < len(template_roi)
                    else (template_roi[-1] if template_roi else 0.0)
                )
                roi = alpha * sim_roi + (1.0 - alpha) * templ_roi

                sim_entropy = (
                    entropy_hist[i - 1]
                    if i - 1 < len(entropy_hist)
                    else (entropy_hist[-1] if entropy_hist else 0.0)
                )
                templ_entropy = (
                    template_entropy[i - 1]
                    if i - 1 < len(template_entropy)
                    else (template_entropy[-1] if template_entropy else 0.0)
                )
                entropy = alpha * sim_entropy + (1.0 - alpha) * templ_entropy
                sim_risk = (
                    risk_hist[i - 1]
                    if i - 1 < len(risk_hist)
                    else (
                        risk_hist[-1]
                        if risk_hist
                        else max(0.0, min(1.0, 1.0 - sim_roi))
                    )
                )
                templ_risk = (
                    template_risk[i - 1]
                    if i - 1 < len(template_risk)
                    else (
                        template_risk[-1]
                        if template_risk
                        else max(0.0, min(1.0, 1.0 - templ_roi))
                    )
                )
                risk = alpha * sim_risk + (1.0 - alpha) * templ_risk
                risk = max(0.0, min(1.0, risk))
                confidence = samples / (samples + 1.0)
                decay = entropy * 0.1 * i

                projections.append(
                    CycleProjection(
                        cycle=float(i),
                        roi=float(roi),
                        risk=float(risk),
                        confidence=float(confidence),
                        decay=float(decay),
                    )
                )
            
            roi_vals = [p.roi for p in projections]
            variance = float(np.var(roi_vals)) if roi_vals else 0.0
            forecast_confidence = (samples / (samples + 1.0)) * (1.0 / (1.0 + variance))
        else:
            slope, intercept, stability = self.tracker.get_trend_curve(wf_id)

            hist_len = len(roi_hist) if roi_hist else 1
            for i in range(1, cycles + 1):
                sim_roi = roi_hist[i - 1] if i - 1 < len(roi_hist) else (
                    roi_hist[-1] if roi_hist else 0.0
                )
                trend_roi = intercept + slope * i
                roi = trend_roi + sim_roi * (i / hist_len)

                sim_conf = conf_hist[i - 1] if i - 1 < len(conf_hist) else (
                    conf_hist[-1] if conf_hist else 0.0
                )
                confidence = max(0.0, min(1.0, sim_conf * stability))

                entropy = (
                    entropy_hist[i - 1]
                    if i - 1 < len(entropy_hist)
                    else (entropy_hist[-1] if entropy_hist else 0.0)
                )
                if risk_hist:
                    sim_risk = (
                        risk_hist[i - 1]
                        if i - 1 < len(risk_hist)
                        else (risk_hist[-1] if risk_hist else 0.0)
                    )
                    risk = max(0.0, min(1.0, sim_risk + (1.0 - stability)))
                else:
                    profile = self.tracker.get_temporal_profile(wf_id)
                    resil = [p.get("resilience", 0.0) for p in profile]
                    baseline = float(np.mean(resil)) if resil else 0.0
                    risk = max(0.0, min(1.0, 1.0 - baseline))

                decay = entropy * 0.1 * i

                projections.append(
                    CycleProjection(
                        cycle=float(i),
                        roi=float(roi),
                        risk=float(risk),
                        confidence=float(confidence),
                        decay=float(decay),
                    )
                )

            variance = float(np.var(roi_hist)) if roi_hist else 0.0
            forecast_confidence = (samples / (samples + 1.0)) * (1.0 / (1.0 + variance))

        result = ForecastResult(projections, float(forecast_confidence), upgrade_id)

        # Persist record to disk and optionally log
        try:
            record = {
                "workflow_id": wf_id,
                "patch": patch_repr,
                "upgrade_id": upgrade_id,
                "projections": [p.__dict__ for p in projections],
                "confidence": result.confidence,
                "timestamp": int(time.time()),
            }
            out_path = self.records_base / f"{wf_id}_{patch_hash}.json"
            with out_path.open("w", encoding="utf8") as fh:
                json.dump(record, fh)
            if self.logger is not None:
                try:
                    self.logger.log(record)
                except Exception:
                    pass
        except Exception:
            pass

        return result


def load_record(
    workflow_id: str,
    upgrade_id: str | None = None,
    patch: Iterable[str] | str | None = None,
    records_base: str | Path = "forecast_records",
) -> ForecastResult:
    """Load a persisted forecast record.

    Parameters
    ----------
    workflow_id:
        Identifier of the workflow whose record should be loaded.
    upgrade_id:
        Identifier of the upgrade. When omitted, the most recent record for the
        workflow is returned. If ``patch`` is provided, this value is ignored.
    patch:
        Patch content used to derive the deterministic record hash. When given,
        the hash of this patch is used to locate the record.
    records_base:
        Directory containing forecast records. Defaults to ``"forecast_records"``.

    Returns
    -------
    ForecastResult
        The deserialised forecast result.
    """

    wf_id = str(workflow_id)
    base = Path(records_base)
    if patch is not None:
        patch_repr = list(patch) if not isinstance(patch, str) else patch
        try:
            patch_serial = json.dumps(patch_repr, sort_keys=True)
        except Exception:
            patch_serial = repr(patch_repr)
        upgrade_id = hashlib.sha256(patch_serial.encode("utf8")).hexdigest()

    if upgrade_id is None:
        latest: tuple[int, float] | None = None
        latest_data: dict | None = None
        for path in base.glob(f"{wf_id}_*.json"):
            try:
                with path.open("r", encoding="utf8") as fh:
                    data = json.load(fh)
                ts = int(data.get("timestamp", 0))
                key = (ts, path.stat().st_mtime)
            except Exception:
                continue
            if latest is None or key >= latest:
                latest = key
                latest_data = data
        if latest_data is None:
            raise FileNotFoundError(f"No record found for workflow {wf_id}")
        data = latest_data
    else:
        path = base / f"{wf_id}_{upgrade_id}.json"
        with path.open("r", encoding="utf8") as fh:
            data = json.load(fh)

    projections = [CycleProjection(**p) for p in data.get("projections", [])]
    confidence = float(data.get("confidence", 0.0))
    rec_upgrade_id = str(data.get("upgrade_id", ""))
    return ForecastResult(projections, confidence, rec_upgrade_id)


def list_records(records_base: str | Path) -> List[str]:
    """Return all forecast record file names.

    Parameters
    ----------
    records_base:
        Directory containing forecast record JSON files.

    Returns
    -------
    List[str]
        Sorted list of record file names. Non-JSON files are ignored.
    """

    base = Path(records_base)
    if not base.exists():
        return []
    return sorted(p.name for p in base.glob("*.json") if p.is_file())


def delete_record(
    workflow_id: str,
    upgrade_id: str | None = None,
    records_base: str | Path = "forecast_records",
) -> None:
    """Delete a forecast record.

    Parameters
    ----------
    workflow_id:
        Identifier of the workflow whose record should be removed.
    upgrade_id:
        Identifier of the upgrade. When omitted, the most recent record for the
        workflow is deleted.
    records_base:
        Directory containing forecast records. Defaults to ``"forecast_records"``.
    """

    wf_id = str(workflow_id)
    base = Path(records_base)

    if upgrade_id is None:
        latest: tuple[int, float] | None = None
        latest_path: Path | None = None
        for path in base.glob(f"{wf_id}_*.json"):
            try:
                with path.open("r", encoding="utf8") as fh:
                    data = json.load(fh)
                ts = int(data.get("timestamp", 0))
                key = (ts, path.stat().st_mtime)
            except Exception:
                continue
            if latest is None or key >= latest:
                latest = key
                latest_path = path
        if latest_path is None:
            raise FileNotFoundError(f"No record found for workflow {wf_id}")
        latest_path.unlink()
    else:
        path = base / f"{wf_id}_{upgrade_id}.json"
        path.unlink()


__all__ = [
    "CycleProjection",
    "ForecastResult",
    "UpgradeForecaster",
    "load_record",
    "list_records",
    "delete_record",
]

