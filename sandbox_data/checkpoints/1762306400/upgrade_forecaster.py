"""Project patch impact over forthcoming improvement cycles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import json
import time
import hashlib
import logging
import numpy as np

try:  # pragma: no cover - allow package or local imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback
    from dynamic_path_router import resolve_path  # type: ignore

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
        simulations: int = 3,
    ) -> None:
        self.tracker = tracker
        self.horizon = max(3, min(5, int(horizon)))
        self.records_base = Path(resolve_path(records_base))
        self.records_base.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.simulations = max(3, min(5, int(simulations)))

    def _log_error(
        self,
        event: str,
        wf_id: str,
        patch_summary: object,
        cycles: int,
        exc: Exception,
    ) -> None:
        """Log an exception via the provided logger or the std logging module."""
        record = {
            "event": event,
            "workflow_id": wf_id,
            "patch": patch_summary,
            "cycles": cycles,
            "error": repr(exc),
        }
        if self.logger is not None:
            try:
                self.logger.log(record)
            except Exception:  # pragma: no cover - best effort
                logging.exception("forecast logging failed: %s", event)
        else:
            logging.error(
                "%s wf=%s patch=%s cycles=%s error=%s",
                event,
                wf_id,
                patch_summary if isinstance(patch_summary, str) else repr(patch_summary),
                cycles,
                exc,
            )

    # ------------------------------------------------------------------
    def forecast(
        self,
        workflow_id: str,
        patch: Iterable[str] | str,
        cycles: int | None = None,
        simulations: int | None = None,
    ) -> ForecastResult:
        """Return cycle projections and overall forecast confidence."""
        cycles = self.horizon if cycles is None else max(3, min(5, int(cycles)))
        simulations = (
            self.simulations
            if simulations is None
            else max(3, min(5, int(simulations)))
        )
        wf_id = str(workflow_id)

        patch_repr = list(patch) if not isinstance(patch, str) else patch
        try:
            patch_serial = json.dumps(patch_repr, sort_keys=True)
        except Exception:
            patch_serial = repr(patch_repr)
        patch_hash = hashlib.sha1(patch_serial.encode("utf8")).hexdigest()
        upgrade_id = patch_hash

        # Simulate the patched workflow multiple times to obtain prospective metrics
        roi_runs: List[List[float]] = []
        conf_runs: List[List[float]] = []
        entropy_runs: List[List[float]] = []
        risk_runs: List[List[float]] = []
        for _ in range(simulations):
            try:
                roi_tracker = simulate_temporal_trajectory(
                    wf_id, patch, foresight_tracker=self.tracker
                )
                roi_runs.append(getattr(roi_tracker, "roi_history", []) or [])
                conf_runs.append(
                    getattr(roi_tracker, "confidence_history", []) or []
                )
                metrics_hist = getattr(roi_tracker, "metrics_history", {}) or {}
                entropy_runs.append(
                    metrics_hist.get("synergy_shannon_entropy", []) or []
                )
                risk_runs.append(metrics_hist.get("synergy_risk_index", []) or [])
            except Exception as exc:  # pragma: no cover - exercised in tests
                self._log_error(
                    "simulate_temporal_trajectory_failed",
                    wf_id,
                    patch_repr,
                    cycles,
                    exc,
                )

        samples = len(self.tracker.history.get(wf_id, []))
        cold_start = self.tracker.is_cold_start(wf_id)

        def _pad(runs: List[List[float]]) -> List[List[float]]:
            return [
                (r[:cycles] + [r[-1]] * (cycles - len(r))) if r else [0.0] * cycles
                for r in runs
            ]

        if roi_runs:
            roi_pad = _pad(roi_runs)
            roi_means = list(np.mean(roi_pad, axis=0))
            roi_vars = list(np.var(roi_pad, axis=0))
        else:
            roi_means = []
            roi_vars = []
        if conf_runs:
            conf_pad = _pad(conf_runs)
            conf_means = list(np.mean(conf_pad, axis=0))
        else:
            conf_means = []
        if entropy_runs:
            entropy_pad = _pad(entropy_runs)
            entropy_means = list(np.mean(entropy_pad, axis=0))
        else:
            entropy_means = []
        if risk_runs:
            risk_pad = _pad(risk_runs)
            risk_means = list(np.mean(risk_pad, axis=0))
            risk_hist_present = any(r for r in risk_runs)
        else:
            risk_means = []
            risk_hist_present = False

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

            if not template_roi or not template_entropy or not template_risk:
                fetch_cssm = getattr(self.tracker, "fetch_cssm_templates", None)
                if callable(fetch_cssm):
                    try:
                        cssm_templates = fetch_cssm(wf_id) or {}
                    except Exception:
                        cssm_templates = {}
                    if not template_roi:
                        template_roi = list(cssm_templates.get("roi") or [])
                    if not template_entropy:
                        template_entropy = list(cssm_templates.get("entropy") or [])
                    if not template_risk:
                        template_risk = list(cssm_templates.get("risk") or [])

            alpha = min(1.0, samples / 5.0)
            variance_terms: List[float] = []
            for i in range(1, cycles + 1):
                sim_roi = (
                    roi_means[i - 1]
                    if i - 1 < len(roi_means)
                    else (roi_means[-1] if roi_means else 0.0)
                )
                templ_roi = (
                    template_roi[i - 1]
                    if i - 1 < len(template_roi)
                    else (template_roi[-1] if template_roi else 0.0)
                )
                roi = alpha * sim_roi + (1.0 - alpha) * templ_roi
                sim_var = (
                    roi_vars[i - 1]
                    if i - 1 < len(roi_vars)
                    else (roi_vars[-1] if roi_vars else 0.0)
                )
                variance_terms.append((alpha ** 2) * sim_var)

                sim_entropy = (
                    entropy_means[i - 1]
                    if i - 1 < len(entropy_means)
                    else (entropy_means[-1] if entropy_means else 0.0)
                )
                templ_entropy = (
                    template_entropy[i - 1]
                    if i - 1 < len(template_entropy)
                    else (template_entropy[-1] if template_entropy else 0.0)
                )
                entropy = alpha * sim_entropy + (1.0 - alpha) * templ_entropy
                if risk_hist_present:
                    sim_risk = (
                        risk_means[i - 1]
                        if i - 1 < len(risk_means)
                        else (risk_means[-1] if risk_means else 0.0)
                    )
                else:
                    sim_risk = max(0.0, min(1.0, 1.0 - sim_roi))
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
            cycle_variance = float(np.var(roi_vals)) if roi_vals else 0.0
            sim_variance = float(np.mean(variance_terms)) if variance_terms else 0.0
            variance = cycle_variance + sim_variance
            forecast_confidence = (samples / (samples + 1.0)) * (1.0 / (1.0 + variance))
        else:
            slope, intercept, stability = self.tracker.get_trend_curve(wf_id)

            hist_len = len(roi_means) if roi_means else 1
            if risk_hist_present:
                baseline_risk = 0.0
                template_risk: List[float] = []
                alpha = 0.0
            else:
                profile = self.tracker.get_temporal_profile(wf_id)
                resil = [p.get("resilience", 0.0) for p in profile]
                baseline = float(np.mean(resil)) if resil else 0.0
                baseline_risk = max(0.0, min(1.0, 1.0 - baseline))
                get_risk_template = getattr(
                    self.tracker, "get_risk_template_curve", None
                )
                template_risk = []
                if callable(get_risk_template):
                    try:
                        template_risk = list(get_risk_template(wf_id) or [])
                    except Exception:
                        template_risk = []
                alpha = min(1.0, samples / 5.0)

            variance_terms: List[float] = []
            for i in range(1, cycles + 1):
                sim_roi = (
                    roi_means[i - 1]
                    if i - 1 < len(roi_means)
                    else (roi_means[-1] if roi_means else 0.0)
                )
                trend_roi = intercept + slope * i
                roi = trend_roi + sim_roi * (i / hist_len)

                sim_var = (
                    roi_vars[i - 1]
                    if i - 1 < len(roi_vars)
                    else (roi_vars[-1] if roi_vars else 0.0)
                )
                variance_terms.append(((i / hist_len) ** 2) * sim_var)

                sim_conf = (
                    conf_means[i - 1]
                    if i - 1 < len(conf_means)
                    else (conf_means[-1] if conf_means else 0.0)
                )
                confidence = max(0.0, min(1.0, sim_conf * stability))

                entropy = (
                    entropy_means[i - 1]
                    if i - 1 < len(entropy_means)
                    else (entropy_means[-1] if entropy_means else 0.0)
                )
                if risk_hist_present:
                    sim_risk = (
                        risk_means[i - 1]
                        if i - 1 < len(risk_means)
                        else (risk_means[-1] if risk_means else 0.0)
                    )
                    risk = max(0.0, min(1.0, sim_risk + (1.0 - stability)))
                else:
                    templ_risk = (
                        template_risk[i - 1]
                        if i - 1 < len(template_risk)
                        else (template_risk[-1] if template_risk else baseline_risk)
                    )
                    risk = alpha * baseline_risk + (1.0 - alpha) * templ_risk
                    risk = max(0.0, min(1.0, risk))

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
            cycle_variance = float(np.var(roi_vals)) if roi_vals else 0.0
            sim_variance = float(np.mean(variance_terms)) if variance_terms else 0.0
            variance = cycle_variance + sim_variance
            forecast_confidence = (samples / (samples + 1.0)) * (1.0 / (1.0 + variance))

        result = ForecastResult(projections, float(forecast_confidence), upgrade_id)

        # Persist record to disk and optionally log
        record = {
            "workflow_id": wf_id,
            "patch": patch_repr,
            "upgrade_id": upgrade_id,
            "projections": [p.__dict__ for p in projections],
            "confidence": result.confidence,
            "timestamp": int(time.time()),
        }
        out_path = self.records_base / f"{wf_id}_{patch_hash}.json"
        try:
            with out_path.open("w", encoding="utf8") as fh:
                json.dump(record, fh)
        except Exception as exc:  # pragma: no cover - exercised in tests
            self._log_error(
                "record_write_failed", wf_id, patch_repr, cycles, exc
            )
        else:
            legacy_path = self.records_base / f"{wf_id}.json"
            try:
                if legacy_path.exists():
                    legacy_path.unlink()
            except Exception as exc:  # pragma: no cover - exercised in tests
                self._log_error(
                    "record_cleanup_failed", wf_id, patch_repr, cycles, exc
                )

        if self.logger is not None:
            try:
                self.logger.log(record)
            except Exception as exc:  # pragma: no cover - best effort
                self._log_error("record_log_failed", wf_id, patch_repr, cycles, exc)

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
    base = Path(resolve_path(records_base))
    if patch is not None:
        patch_repr = list(patch) if not isinstance(patch, str) else patch
        try:
            patch_serial = json.dumps(patch_repr, sort_keys=True)
        except Exception:
            patch_serial = repr(patch_repr)
        upgrade_id = hashlib.sha1(patch_serial.encode("utf8")).hexdigest()

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
            legacy = base / f"{wf_id}.json"
            if not legacy.exists():
                raise FileNotFoundError(f"No record found for workflow {wf_id}")
            with legacy.open("r", encoding="utf8") as fh:
                data = json.load(fh)
        else:
            data = latest_data
    else:
        path = base / f"{wf_id}_{upgrade_id}.json"
        with path.open("r", encoding="utf8") as fh:
            data = json.load(fh)

    projections = [CycleProjection(**p) for p in data.get("projections", [])]
    confidence = float(data.get("confidence", 0.0))
    rec_upgrade_id = str(data.get("upgrade_id", ""))
    return ForecastResult(projections, confidence, rec_upgrade_id)


def list_records(records_base: str | Path) -> List[dict]:
    """Return metadata for all forecast records.

    Parameters
    ----------
    records_base:
        Directory containing forecast record JSON files.

    Returns
    -------
    List[dict]
        Each element contains ``workflow_id``, ``upgrade_id`` and ``timestamp``
        keys describing a persisted forecast record. Invalid files are ignored
        and the result is sorted by workflow and timestamp.
    """

    base = Path(resolve_path(records_base))
    if not base.exists():
        return []

    records: List[dict] = []
    for path in base.glob("*.json"):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf8") as fh:
                data = json.load(fh)
            record = {
                "workflow_id": str(data.get("workflow_id", "")),
                "upgrade_id": str(data.get("upgrade_id", "")),
                "timestamp": int(data.get("timestamp", 0)),
            }
        except Exception:
            continue
        records.append(record)

    return sorted(records, key=lambda r: (r["workflow_id"], r["timestamp"]))


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
    base = Path(resolve_path(records_base))

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
            legacy = base / f"{wf_id}.json"
            if not legacy.exists():
                raise FileNotFoundError(f"No record found for workflow {wf_id}")
            legacy.unlink()
        else:
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

