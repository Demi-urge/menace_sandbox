"""Track workflow metrics and forecast ROI collapse risk.

`ForesightTracker` stores recent ROI and related metrics for each workflow and
offers :meth:`predict_roi_collapse` to extrapolate these trends. The helper
adjusts the ROI slope using recorded entropy degradation and volatility to
estimate when the trajectory will dip below zero.

``predict_roi_collapse`` returns a dictionary with a projected ROI ``curve``,
an optional ``collapse_in`` horizon and two diagnostic fields:

``risk``
    One of ``Stable``, ``Slow decay``, ``Volatile`` or ``Immediate collapse
    risk`` depending on the slope and volatility of recent ROI deltas.
``brittle``
    ``True`` when minor entropy increases trigger large ROI drops.

Baseline curves can be supplied through ``configs/foresight_templates.yaml``.
The file supports ``profiles`` and ``trajectories`` for ROI bootstrapping and
optional ``entropy_profiles``/``risk_profiles`` with matching
``entropy_trajectories``/``risk_trajectories``.

Example
-------
>>> from menace_sandbox.foresight_tracker import ForesightTracker
>>> tracker = ForesightTracker()
>>> info = tracker.predict_roi_collapse("workflow-1")
>>> info["risk"]
'Stable'
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Tuple, TYPE_CHECKING
from numbers import Number
import logging
import math

from dependency_health import (
    dependency_registry,
    DependencyCategory,
    DependencySeverity,
)

try:  # pragma: no cover - optional dependency
    import numpy as np
    _NUMPY_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - executed without numpy
    np = None  # type: ignore[assignment]
    _NUMPY_ERROR = exc
else:  # pragma: no cover - keep annotation when numpy available
    _NUMPY_ERROR = None

import yaml
import os

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

if _NUMPY_ERROR is not None:  # pragma: no cover - informational log
    dependency_registry.mark_missing(
        name="numpy",
        category=DependencyCategory.PYTHON,
        optional=True,
        severity=DependencySeverity.INFO,
        description="High-precision numerical routines for foresight tracking",
        reason=str(_NUMPY_ERROR),
        remedy="pip install numpy",
        logger=logger,
    )
else:
    dependency_registry.mark_available(
        name="numpy",
        category=DependencyCategory.PYTHON,
        optional=True,
        description="High-precision numerical routines for foresight tracking",
        logger=logger,
    )


def _std(values: List[float], ddof: int = 0) -> float:
    n = len(values)
    if n == 0 or n <= ddof:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values)
    denom = n - ddof if ddof else n
    if denom <= 0:
        return 0.0
    return math.sqrt(variance / denom)


def _linear_slope(x: List[int], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    denom = sum((xi - mean_x) ** 2 for xi in x)
    if denom == 0:
        return 0.0
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return num / denom


def _finite_second_derivative(values: List[float]) -> float:
    if len(values) < 3:
        return 0.0
    return values[-1] - 2.0 * values[-2] + values[-3]


def _diff(values: List[float]) -> List[float]:
    return [b - a for a, b in zip(values, values[1:])]

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from roi_tracker import ROITracker


class ForesightTracker:
    """Maintain recent cycle or stage metrics for workflows and evaluate stability.

    Entries in :attr:`history` may represent sequential temporal trajectory
    stages rather than strictly live cycles.
    """

    def __init__(
        self,
        max_cycles: int = 10,
        volatility_threshold: float = 1.0,
        window: int | None = None,
        N: int | None = None,
        templates_path: str | Path = resolve_path("configs/foresight_templates.yaml"),
        template_config_path: str | Path | None = None,
        cssm_client: object | None = None,
    ) -> None:
        """Create a new tracker.

        Parameters
        ----------
        max_cycles, window, N:
            Aliases specifying the number of recent cycles to retain per
            workflow.
        volatility_threshold:
            Maximum standard deviation across stored metrics considered stable.
        templates_path:
            Location of the ROI template configuration. ``template_config_path``
            remains as a deprecated alias for backwards compatibility.
        """

        if window is not None:
            max_cycles = window
        if N is not None:
            max_cycles = N

        self.max_cycles = max_cycles
        self.volatility_threshold = volatility_threshold
        self.history: Dict[str, Deque[Dict[str, float | str]]] = {}
        if template_config_path is not None:
            templates_path = template_config_path
        cfg_path = resolve_path(templates_path)
        self.template_config_path = cfg_path
        self.templates: Dict[str, list[float]] | None = None
        self.entropy_templates: Dict[str, list[float]] | None = None
        self.risk_templates: Dict[str, list[float]] | None = None
        # map workflow identifiers to their template profile
        self.workflow_profiles: Dict[str, str] = {}
        self.entropy_profiles: Dict[str, str] = {}
        self.risk_profiles: Dict[str, str] = {}
        # backward compatibility for callers expecting ``profile_map``
        self.profile_map = self.workflow_profiles
        self.entropy_profile_map = self.entropy_profiles
        self.risk_profile_map = self.risk_profiles
        self.cssm_client = cssm_client
        self._cssm_template_cache: Dict[str, Dict[str, List[float]]] = {}
        self._load_templates()

    # ------------------------------------------------------------------
    @property
    def window(self) -> int:
        """Legacy alias for :attr:`max_cycles`.

        The tracker historically exposed the maximum retention window under
        the name ``window``.  The public attribute :attr:`max_cycles` now holds
        this value directly; this property preserves backwards compatibility.
        """

        return self.max_cycles

    # ------------------------------------------------------------------
    def record_cycle_metrics(
        self,
        workflow_id: str,
        metrics: Mapping[str, float],
        *,
        compute_stability: bool = False,
        **extra_metrics: float | str,
    ) -> None:
        """Append ``metrics`` for ``workflow_id`` and cap history length.

        Parameters
        ----------
        workflow_id:
            Identifier of the workflow whose metrics are being recorded.
        metrics:
            Mapping of metric names to numeric values for the current cycle or
            stage.
        compute_stability:
            When ``True``, the helper immediately evaluates the current
            stability by calling :meth:`get_trend_curve` and stores the
            resulting value under the ``"stability"`` key in the appended
            history entry.
        **extra_metrics:
            Additional fields, such as ``stage`` or ``degradation``, which are
            merged into the stored entry.  Non‑numeric values are stored as is
            and excluded from trend calculations.
        """

        entry = {k: float(v) for k, v in metrics.items()}
        for key, value in extra_metrics.items():
            entry[key] = float(value) if isinstance(value, Number) else value
        queue = self.history.setdefault(workflow_id, deque(maxlen=self.max_cycles))
        queue.append(entry)
        if compute_stability:
            _, _, stability = self.get_trend_curve(workflow_id)
            entry["stability"] = stability

    # ------------------------------------------------------------------
    def capture_from_roi(
        self,
        tracker: "ROITracker" | None,
        workflow_id: str,
        profile: str | None = None,
        stage: str | None = None,
        compute_stability: bool = False,
    ) -> None:
        """Record metrics extracted from an :class:`ROITracker` instance.

        Parameters
        ----------
        tracker:
            ROI tracker providing ``roi_history``, ``raroi_history``,
            ``confidence_history`` and ``metrics_history``.
        workflow_id:
            Identifier of the workflow for which the metrics should be
            recorded.
        profile:
            Optional profile name used to select ROI templates. When provided,
            it registers ``workflow_id`` under that profile. When omitted, the
            existing mapping is used or the ``workflow_id`` itself if no
            profile has been registered.
        stage:
            Optional label describing the workflow stage for which the metrics
            are being captured.  Forwarded to :meth:`record_cycle_metrics`.
        compute_stability:
            When ``True`` the stability of the resulting entry is computed
            immediately via :meth:`record_cycle_metrics`.

        The helper pulls the latest ROI delta, RAROI delta, resilience,
        confidence, entropy and scenario degradation metrics from ``tracker``
        and forwards them to :meth:`record_cycle_metrics`.
        """

        if tracker is None:
            return

        try:
            roi_hist = getattr(tracker, "roi_history", [])
            raroi_hist = getattr(tracker, "raroi_history", [])
            history = self.history.get(workflow_id)
            if profile is not None:
                self.workflow_profiles[workflow_id] = profile

            real_roi = float(roi_hist[-1]) if roi_hist else 0.0
            if len(raroi_hist) >= 2:
                real_raroi = float(raroi_hist[-1] - raroi_hist[-2])
            elif raroi_hist:
                real_raroi = float(raroi_hist[-1])
            else:
                real_raroi = real_roi

            if self.is_cold_start(workflow_id):
                cycles = len(history) if history else 0
                template_val = self.get_template_value(workflow_id, cycles)
                if template_val is not None:
                    alpha = min(1.0, cycles / 5.0)
                    roi_delta = alpha * real_roi + (1.0 - alpha) * template_val
                    raroi_delta = alpha * real_raroi + (1.0 - alpha) * template_val
                else:
                    roi_delta = real_roi
                    raroi_delta = real_raroi
            else:
                roi_delta = real_roi
                raroi_delta = real_raroi

            conf_hist = getattr(tracker, "confidence_history", [])
            confidence = float(conf_hist[-1]) if conf_hist else 0.0

            metrics_hist = getattr(tracker, "metrics_history", {})
            res_hist = metrics_hist.get("synergy_resilience") or metrics_hist.get(
                "resilience", []
            )
            resilience = float(res_hist[-1]) if res_hist else 0.0

            ent_hist = (
                metrics_hist.get("synergy_shannon_entropy")
                or metrics_hist.get("shannon_entropy")
                or metrics_hist.get("entropy")
                or []
            )
            entropy = float(ent_hist[-1]) if ent_hist else 0.0

            try:
                scenario_deg = float(
                    getattr(tracker, "scenario_degradation", lambda: 0.0)()
                )
            except Exception:
                scenario_deg = 0.0

            self.record_cycle_metrics(
                workflow_id,
                {
                    "roi_delta": roi_delta,
                    "raroi_delta": raroi_delta,
                    "confidence": confidence,
                    "resilience": resilience,
                    "synergy_shannon_entropy": entropy,
                    "scenario_degradation": scenario_deg,
                    "raw_roi_delta": real_roi,
                },
                compute_stability=compute_stability,
                stage=stage,
            )
        except Exception:
            # best effort; failing to record foresight metrics shouldn't break
            # the calling workflow
            pass

    # ------------------------------------------------------------------
    def is_cold_start(self, workflow_id: str) -> bool:
        """Return ``True`` when ``workflow_id`` lacks meaningful ROI history.

        A workflow is considered a *cold start* until at least three cycles
        have been recorded **and** one of those entries contains a non‑zero
        ``roi_delta`` value.  This guards the template blending logic in
        :meth:`capture_from_roi` which needs genuine ROI feedback to be
        effective.

        Parameters
        ----------
        workflow_id:
            Identifier whose history should be inspected.

        Returns
        -------
        bool
            ``True`` if fewer than three cycles exist or all ``roi_delta``
            values are missing/zero; ``False`` otherwise.
        """

        history = self.history.get(workflow_id)
        if not history or len(history) < 3:
            return True

        has_signal = any(float(entry.get("roi_delta", 0.0)) != 0.0 for entry in history)
        return not has_signal

    # ------------------------------------------------------------------
    def _load_templates(self) -> None:
        """Load ROI, entropy, and risk template curves from disk.

        The configuration may define ``profiles`` and ``trajectories`` (or
        ``templates``) for ROI. Optional ``entropy_profiles`` and
        ``entropy_trajectories`` provide sample entropy degradation curves,
        while ``risk_profiles`` and ``risk_trajectories`` allow basic risk
        comparisons. All sections are loaded lazily on first access.
        """

        if self.templates is not None and self.entropy_templates is not None:
            return
        try:
            with self.template_config_path.open("r", encoding="utf8") as fh:
                data = yaml.safe_load(fh) or {}

            # ROI templates -------------------------------------------------
            profiles = data.get("profiles")
            trajectories = data.get("trajectories") or data.get("templates")

            # Optional entropy templates for degradation analysis ---------
            ent_profiles = data.get("entropy_profiles")
            ent_traj = data.get("entropy_trajectories") or data.get(
                "entropy_templates"
            )

            # Optional risk templates -------------------------------------
            risk_profiles = data.get("risk_profiles")
            risk_traj = data.get("risk_trajectories") or data.get(
                "risk_templates"
            )

            merged: Dict[str, list[float]] = {}
            merged_entropy: Dict[str, list[float]] = {}
            merged_risk: Dict[str, list[float]] = {}
            if isinstance(trajectories, Mapping):
                for name, curve in trajectories.items():
                    merged[str(name)] = [float(v) for v in curve]
            if isinstance(ent_traj, Mapping):
                for name, curve in ent_traj.items():
                    merged_entropy[str(name)] = [float(v) for v in curve]
            if isinstance(risk_traj, Mapping):
                for name, curve in risk_traj.items():
                    merged_risk[str(name)] = [float(v) for v in curve]

            if isinstance(profiles, Mapping):
                for wf_id, prof in profiles.items():
                    self.workflow_profiles[str(wf_id)] = str(prof)
                    if ent_profiles is None:
                        self.entropy_profiles.setdefault(str(wf_id), str(prof))
            if isinstance(ent_profiles, Mapping):
                for wf_id, prof in ent_profiles.items():
                    self.entropy_profiles[str(wf_id)] = str(prof)
            if isinstance(risk_profiles, Mapping):
                for wf_id, prof in risk_profiles.items():
                    self.risk_profiles[str(wf_id)] = str(prof)

            self.templates = merged
            self.entropy_templates = merged_entropy
            self.risk_templates = merged_risk
        except Exception:
            self.templates = {}
            self.entropy_templates = {}
            self.risk_templates = {}

    # ------------------------------------------------------------------
    def get_template_curve(self, workflow_id: str) -> list[float]:
        """Return the ROI template curve for ``workflow_id``.

        The templates are loaded lazily on first access.  The workflow's
        registered profile determines which trajectory is returned. If the
        workflow has no registered profile, its own identifier is used. An
        empty list is returned when no matching trajectory exists.
        """

        self._load_templates()
        profile = self.workflow_profiles.get(workflow_id, workflow_id)
        curve = self.templates.get(profile, []) if self.templates else []
        return [float(v) for v in curve]

    # ------------------------------------------------------------------
    def get_entropy_template_curve(self, workflow_id: str) -> list[float]:
        """Return the entropy template curve for ``workflow_id``.

        This mirrors :meth:`get_template_curve` but looks up the trajectory for
        the ``synergy_shannon_entropy`` metric.  When no matching curve exists,
        an empty list is returned.  Profile mappings fall back to the ROI
        mappings when dedicated entropy profiles are absent.
        """

        self._load_templates()
        profile = self.entropy_profiles.get(
            workflow_id,
            self.workflow_profiles.get(workflow_id, workflow_id),
        )
        curve = (
            self.entropy_templates.get(profile, [])
            if self.entropy_templates
            else []
        )
        return [float(v) for v in curve]

    # ------------------------------------------------------------------
    def get_risk_template_curve(self, workflow_id: str) -> list[float]:
        """Return the risk template curve for ``workflow_id``.

        Risk templates mirror the structure of ROI templates but represent the
        anticipated ``synergy_risk_index`` trajectory.  When no dedicated risk
        profile or trajectory exists, an empty list is returned.
        """

        self._load_templates()
        profile = self.risk_profiles.get(
            workflow_id, self.workflow_profiles.get(workflow_id, workflow_id)
        )
        curve = (
            self.risk_templates.get(profile, []) if self.risk_templates else []
        )
        return [float(v) for v in curve]

    # ------------------------------------------------------------------
    def fetch_cssm_templates(self, workflow_id: str) -> Dict[str, List[float]]:
        """Retrieve ROI, entropy and risk templates from CSSM.

        Results are cached per ``workflow_id``. When ``cssm_client`` is
        provided it is invoked with ``workflow_id`` and should return a mapping
        with ``roi``, ``entropy`` and ``risk`` lists. Otherwise an HTTP
        request is performed against ``CSSM_URL``.
        """

        cache = self._cssm_template_cache
        if workflow_id in cache:
            return cache[workflow_id]

        data: Mapping[str, List[float]] | None = None
        client = getattr(self, "cssm_client", None)
        if callable(client):
            try:
                data = client(workflow_id)
            except Exception:
                data = None

        if data is None and requests:
            url = os.getenv("CSSM_URL")
            if url:
                try:
                    resp = requests.get(f"{url.rstrip('/')}/{workflow_id}", timeout=5)
                    if resp.status_code == 200:
                        payload = resp.json()
                        if isinstance(payload, Mapping):
                            data = payload
                except Exception:  # pragma: no cover - best effort
                    data = None

        if not data:
            data = {}

        templates = {
            "roi": [float(v) for v in data.get("roi", [])],
            "entropy": [float(v) for v in data.get("entropy", [])],
            "risk": [float(v) for v in data.get("risk", [])],
        }
        cache[workflow_id] = templates
        return templates

    # ------------------------------------------------------------------
    def get_template_value(self, workflow_id: str, cycle_index: int) -> float | None:
        """Return the template ROI value for ``workflow_id`` at ``cycle_index``.

        ``None`` is returned when the workflow has no associated template
        trajectory.  Indices beyond the available curve fall back to the last
        template value.
        """

        curve = self.get_template_curve(workflow_id)
        if not curve:
            return None
        if 0 <= cycle_index < len(curve):
            return float(curve[cycle_index])
        return float(curve[-1])

    # ------------------------------------------------------------------
    def get_trend_curve(self, workflow_id: str) -> Tuple[float, float, float]:
        """Return slope, second derivative and average window stability.

        This is part of the public API so callers can inspect the raw trend
        information used by :meth:`is_stable`. The trend is computed from the
        mean of the metric values for each recorded cycle.
        ``avg_window_stability`` equals ``1 / (1 + std)`` where ``std`` is the
        standard deviation over the retained window.
        """

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return 0.0, 0.0, 0.0

        averages: List[float] = []
        for entry in data:
            numeric_vals = [
                float(v) for v in entry.values() if isinstance(v, Number)
            ]
            if np is not None:
                averages.append(float(np.mean(numeric_vals)) if numeric_vals else 0.0)
            else:
                averages.append(
                    sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0.0
                )

        if np is not None:
            arr = np.array(averages, dtype=float)
            x = np.arange(len(arr))
            slope = float(np.polyfit(x, arr, 1)[0])
            if len(arr) >= 3:
                coeff = np.polyfit(x, arr, 2)[0]
                second_derivative = float(2.0 * coeff)
            else:
                second_derivative = 0.0
            window = min(self.max_cycles, len(arr))
            std = float(np.std(arr[-window:], ddof=1)) if window > 1 else 0.0
        else:
            x_vals = list(range(len(averages)))
            slope = _linear_slope(x_vals, averages)
            second_derivative = _finite_second_derivative(averages)
            window = min(self.max_cycles, len(averages))
            std = _std(averages[-window:], ddof=1) if window > 1 else 0.0
        avg_stability = 1.0 / (1.0 + std)
        return slope, second_derivative, avg_stability

    # ------------------------------------------------------------------
    def is_stable(self, workflow_id: str) -> bool:
        """Return ``True`` when slope is positive and volatility is low.

        This public helper combines :meth:`get_trend_curve` with a volatility
        check to determine whether ``workflow_id`` is operating within the
        allowed threshold.
        """

        data = self.history.get(workflow_id)
        if not data or len(data) < 2:
            return False

        slope, _, _ = self.get_trend_curve(workflow_id)
        values = [
            float(v)
            for entry in data
            for v in entry.values()
            if isinstance(v, Number)
        ]
        if np is not None:
            all_values = np.array(values, dtype=float)
            std = float(np.std(all_values, ddof=1)) if all_values.size > 1 else 0.0
        else:
            std = _std(values, ddof=1) if len(values) > 1 else 0.0
        return slope > 0 and std < self.volatility_threshold

    # ------------------------------------------------------------------
    def predict_roi_collapse(self, workflow_id: str, horizon: int = 5) -> Dict[str, object]:
        """Project the ROI trajectory and classify collapse risk.

        The helper inspects recent ``roi_delta`` history for ``workflow_id`` and
        extrapolates the current trend ``horizon`` cycles into the future.  The
        slope is obtained from :func:`get_trend_curve` while cycle‑to‑cycle
        volatility is measured as the standard deviation of consecutive
        ``roi_delta`` changes. Recorded ``scenario_degradation`` is compared to
        :func:`get_entropy_template_curve` and both degradation and volatility
        reduce the projected slope.  When small degradation increases produce
        sharp ROI drops the workflow is flagged as brittle.

        Risk labels follow these guidelines:

        ``Stable``
            slope ≥ 0 and volatility below ``volatility_threshold``.
        ``Slow decay``
            mildly negative slope with low volatility.
        ``Volatile``
            volatility ≥ ``volatility_threshold`` regardless of slope.
        ``Immediate collapse risk``
            steep negative slope or projected failure within a few cycles.

        Parameters
        ----------
        workflow_id:
            Identifier of the workflow to analyse.
        horizon:
            Number of future cycles to project. Defaults to ``5``.

        Returns
        -------
        dict
            Mapping with the projected ROI ``curve`` and diagnostic fields.
            The key ``cycles_to_collapse`` reports the projected time until ROI
            reaches zero (or ``None`` if no collapse is foreseen).  The
            ``risk`` field holds the textual risk category and ``brittle``
            indicates high sensitivity to entropy.  A backward compatible
            ``collapse_in`` alias is also included.
        """

        history = self.history.get(workflow_id)
        if not history:
            return {
                "curve": [],
                "cycles_to_collapse": None,
                "collapse_in": None,
                "risk": "Stable",
                "brittle": False,
            }

        slope, second_derivative, _ = self.get_trend_curve(workflow_id)

        roi_values = [float(e.get("roi_delta", 0.0)) for e in history]
        if len(roi_values) > 1:
            if np is not None:
                volatility = float(np.std(np.diff(roi_values), ddof=1))
            else:
                volatility = _std(_diff(roi_values), ddof=1)
        else:
            volatility = 0.0

        last_entry = history[-1]
        last_roi = float(last_entry.get("roi_delta", 0.0))
        last_deg = float(last_entry.get("scenario_degradation", 0.0))
        cycle_index = len(history) - 1

        ent_curve = self.get_entropy_template_curve(workflow_id)
        if ent_curve:
            template_ent = (
                ent_curve[cycle_index]
                if cycle_index < len(ent_curve)
                else ent_curve[-1]
            )
        else:
            template_ent = 0.0
        entropy_deviation = last_deg - template_ent
        effective_slope = (slope - entropy_deviation) / (1.0 + volatility)

        projection: List[float] = []
        proj_roi = last_roi
        cycles_to_collapse: float | None = 0 if proj_roi <= 0 else None
        for i in range(1, horizon + 1):
            proj_roi += effective_slope
            projection.append(proj_roi)
            if cycles_to_collapse is None and proj_roi <= 0:
                cycles_to_collapse = float(i)

        brittle = False
        if len(history) >= 2:
            prev = history[-2]
            roi_drop = float(prev.get("roi_delta", 0.0)) - last_roi
            deg_diff = last_deg - float(prev.get("scenario_degradation", 0.0))
            if 0 < deg_diff < 0.05 and roi_drop / (deg_diff + 1e-6) > 10.0:
                brittle = True

        if volatility >= self.volatility_threshold:
            risk = "Volatile"
        elif slope >= 0:
            risk = "Stable"
        else:
            steep = slope < -1.0 or second_derivative < -0.1
            if (cycles_to_collapse is not None and cycles_to_collapse <= 2) or steep:
                risk = "Immediate collapse risk"
            else:
                risk = "Slow decay"

        return {
            "curve": projection,
            "cycles_to_collapse": cycles_to_collapse,
            # backward compatible alias
            "collapse_in": cycles_to_collapse,
            "risk": risk,
            "brittle": brittle,
        }

    # ------------------------------------------------------------------
    def get_temporal_profile(self, workflow_id: str) -> List[Dict[str, float]]:
        """Return stored metric entries for ``workflow_id`` in chronological order."""

        history = self.history.get(workflow_id)
        if not history:
            return []
        return [dict(entry) for entry in history]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON‑serialisable representation of the tracker.

        Only the retained window for each workflow is serialised.  History is
        emitted in chronological order so that the first element of each list is
        the oldest entry.
        """

        return {
            "window": self.max_cycles,
            "volatility_threshold": self.volatility_threshold,
            "history": {
                wf_id: [
                    {
                        k: (float(v) if isinstance(v, Number) else v)
                        for k, v in entry.items()
                    }
                    for entry in list(entries)
                ]
                for wf_id, entries in self.history.items()
            },
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: Mapping,
        N: int | None = None,
        volatility_threshold: float | None = None,
        max_cycles: int | None = None,
    ) -> "ForesightTracker":
        """Reconstruct a tracker from :meth:`to_dict` output.

        Parameters
        ----------
        data:
            Mapping produced by :meth:`to_dict`.
        N:
            Optional override for the number of history entries to retain per
            workflow.  When ``None`` the value stored in ``data`` is used or the
            class default (``10``) if unavailable.
        volatility_threshold:
            Optional override for the volatility threshold.  When ``None`` the
            value stored in ``data`` is used or the class default (``1.0``) if
            missing.
        """

        if max_cycles is not None and N is None:
            N = max_cycles
        if N is None:
            N = int(data.get("window", data.get("N", 10)))
        if volatility_threshold is None:
            volatility_threshold = float(data.get("volatility_threshold", 1.0))

        tracker = cls(max_cycles=N, volatility_threshold=volatility_threshold)

        raw_history = data.get("history", {})
        for wf_id, entries in raw_history.items():
            queue: Deque[Dict[str, float]] = deque(maxlen=tracker.max_cycles)
            for entry in list(entries)[-tracker.max_cycles:]:
                queue.append(
                    {
                        k: (float(v) if isinstance(v, Number) else v)
                        for k, v in entry.items()
                    }
                )
            tracker.history[wf_id] = queue

        return tracker


__all__ = ["ForesightTracker"]
