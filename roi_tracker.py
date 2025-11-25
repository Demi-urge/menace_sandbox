from __future__ import annotations

"""Track ROI deltas across self-improvement iterations.

The module exposes :func:`calculate_raroi` which converts a raw ROI into a
risk-adjusted ROI (RAROI). RAROI applies catastrophic risk, recent stability
and critical safety test modifiers::

    raroi = base_roi * (1 - catastrophic_risk)
            * stability_factor * safety_factor

``catastrophic_risk`` multiplies rollback probability by workflow impact
severity. Default severities live in ``config/impact_severity.yaml`` and may be
overridden by setting the ``IMPACT_SEVERITY_CONFIG`` environment variable to a
custom YAML mapping. ``stability_factor`` declines as recent ROI deltas grow
more volatile. Failing tests from critical suites (e.g. security, alignment)
halve ``safety_factor`` so RAROI drops below the raw ROI when risk rises. RAROI close to the raw ROI
indicates a stable, low-risk workflow; a large gap signals substantial
operational risk.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Iterable, Sequence, Mapping, Callable
from dataclasses import dataclass

import argparse
import csv

import types
import json
import os
import sqlite3
import sys
from pathlib import Path
from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES
from dynamic_path_router import get_project_root, resolve_path
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    package_name = package_root.name
    pkg = sys.modules.get(package_name)
    if pkg is None:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(package_root)]
        sys.modules[package_name] = pkg
    __package__ = package_name

try:  # pragma: no cover - allow running as script without package context
    from .logging_utils import get_logger, log_record
    from .config_loader import (
        get_impact_severity,
        impact_severity_map as load_impact_severity_map,
    )
    from .workflow_run_summary import record_run, save_all_summaries
except ImportError:  # pragma: no cover - fallback when executed directly
    from logging_utils import get_logger, log_record  # type: ignore
    from config_loader import (  # type: ignore
        get_impact_severity,
        impact_severity_map as load_impact_severity_map,
    )
    from workflow_run_summary import record_run, save_all_summaries  # type: ignore

try:  # pragma: no cover - telemetry optional during tests
    from .telemetry_backend import TelemetryBackend
    from . import telemetry_backend as tb
except Exception:  # pragma: no cover - provide stub when unavailable
    class TelemetryBackend:  # type: ignore
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def record(self, *a: Any, **k: Any) -> None:
            pass

    tb = None  # type: ignore
try:  # pragma: no cover - support package and script imports
    from .borderline_bucket import BorderlineBucket
    from .truth_adapter import TruthAdapter
    from .roi_calculator import ROICalculator, propose_fix
    from .readiness_index import compute_readiness
except ImportError:  # pragma: no cover - fallback when executed directly
    from borderline_bucket import BorderlineBucket  # type: ignore
    from truth_adapter import TruthAdapter  # type: ignore
    from roi_calculator import ROICalculator, propose_fix  # type: ignore
    from readiness_index import compute_readiness  # type: ignore
try:  # pragma: no cover - optional during tests
    from .roi_results_db import ROIResultsDB  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed directly
    from roi_results_db import ROIResultsDB  # type: ignore
except Exception:  # pragma: no cover - allow operation without DB
    ROIResultsDB = None  # type: ignore
LOCAL_TABLES.add("workflow_module_deltas")

from importlib import import_module
from types import ModuleType

_BOOTSTRAP_WATCHDOG_ENV = "BOOTSTRAP_WATCHDOG_ACTIVE"

_SELF_TEST_SERVICE: ModuleType | None = None
_SELF_TEST_SERVICE_FAILED = False


def _get_self_test_service() -> ModuleType | None:
    """Return the lazily imported :mod:`self_test_service` module."""

    global _SELF_TEST_SERVICE, _SELF_TEST_SERVICE_FAILED

    if _SELF_TEST_SERVICE is not None or _SELF_TEST_SERVICE_FAILED:
        return _SELF_TEST_SERVICE

    candidates = []
    if __package__:
        candidates.append(f"{__package__}.self_test_service")
    candidates.append("self_test_service")

    for name in candidates:
        try:
            module = import_module(name)
        except Exception:
            continue
        _SELF_TEST_SERVICE = module
        return module

    _SELF_TEST_SERVICE_FAILED = True
    return None
try:  # pragma: no cover - optional dependency
    from . import metrics_exporter as _me
except ImportError:  # pragma: no cover - fallback when executed directly
    import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - best effort
    _me = None

if _me is not None:  # pragma: no cover - metrics optional
    try:
        _DB_ROI_GAUGE = _me.Gauge(
            "db_roi_contribution",
            "Average ROI contribution per origin database",
            ["origin_db"],
        )
        _TA_LOW_CONF_GAUGE = _me.Gauge(
            "truth_adapter_low_confidence",
            "TruthAdapter low confidence flag",
        )
        _ROI_BOTTLENECK_GAUGE = _me.Gauge(
            "roi_bottleneck_suggestion",
            "ROI bottleneck suggestion count",
            ["metric", "hint"],
        )
    except Exception:  # pragma: no cover - gauge already registered
        try:
            from prometheus_client import REGISTRY  # type: ignore

            _DB_ROI_GAUGE = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
                "db_roi_contribution"
            )
            _TA_LOW_CONF_GAUGE = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
                "truth_adapter_low_confidence"
            )
            _ROI_BOTTLENECK_GAUGE = REGISTRY._names_to_collectors.get(
                "roi_bottleneck_suggestion"
            )
        except Exception:
            _DB_ROI_GAUGE = None
            _TA_LOW_CONF_GAUGE = None
    _ROI_BOTTLENECK_GAUGE = None
else:  # pragma: no cover - metrics optional
    _DB_ROI_GAUGE = None
    _TA_LOW_CONF_GAUGE = None
    _ROI_BOTTLENECK_GAUGE = None

logger = get_logger(__name__)

from db_router import GLOBAL_ROUTER, init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope

# Reuse a pre-configured router when available, otherwise create a local one
# for standalone execution.
router = GLOBAL_ROUTER or init_db_router("roi_tracker")


def _resolve_telemetry_db_path() -> str:
    """Return a filesystem path for the telemetry database.

    ``resolve_path`` normally locates existing resources inside the repository
    roots.  When the telemetry database has not been created yet we create an
    empty file inside the primary project root so the ``TelemetryBackend`` can
    bootstrap its schema on first use.  This mirrors the behaviour of other
    SQLite-backed components which initialise their databases lazily.
    """

    try:
        return str(resolve_path("telemetry.db"))
    except FileNotFoundError:
        candidate = get_project_root() / "telemetry.db"
        candidate.touch(exist_ok=True)
        return str(candidate)

# Critical test suites that significantly impact safety if they fail. The
# collection may be extended as the project grows but defaults cover the most
# sensitive areas.
CRITICAL_SUITES = {"security", "alignment"}

# Penalties applied when critical test suites fail.  Values may be overridden
# by setting the ``CRITICAL_TEST_PENALTIES`` environment variable to a JSON
# mapping, e.g. ``{"security": 0.4, "alignment": 0.7}``.
DEFAULT_PENALTY = 0.5
CRITICAL_TEST_PENALTIES: dict[str, float] = {
    name: DEFAULT_PENALTY for name in CRITICAL_SUITES
}
try:  # pragma: no cover - best effort
    _pen = os.getenv("CRITICAL_TEST_PENALTIES")
    if _pen:
        CRITICAL_TEST_PENALTIES.update(
            {str(k).lower(): float(v) for k, v in json.loads(_pen).items()}
        )
except Exception:
    pass

# Recommendations to harden scenarios that show weaknesses.
HARDENING_TIPS: dict[str, str] = {
    "concurrency_spike": "add locking or queueing",
    "schema_drift": "tighten schema validation",
    "hostile_input": "sanitize inputs",
    "flaky_upstream": "add retries or fallback logic",
}

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .resources_bot import ROIHistoryDB
    from .prediction_manager_bot import PredictionManager
    from .vector_metrics_db import VectorMetricsDB


def _estimate_rollback_probability(metrics: Mapping[str, float]) -> float:
    """Return the chance that a workflow will require rollback.

    The helper interprets a variety of runtime metrics and normalises them into
    a probability between 0 and 1.  Recognised metrics include
    ``errors_per_minute`` (scaled by ``error_threshold"), ``test_flakiness`` or
    ``flaky_tests`` counts (scaled by ``flakiness_threshold"), direct
    ``recent_failure_rate``/``failure_rate`` values and an ``instability`` score.
    Unrecognised metrics are ignored.
    """

    if not metrics:
        return 0.0

    errors_per_minute = float(metrics.get("errors_per_minute", 0.0))
    error_threshold = float(metrics.get("error_threshold", 10.0))
    error_prob = (
        errors_per_minute / error_threshold if error_threshold > 0 else 0.0
    )

    flakiness = float(metrics.get("test_flakiness", metrics.get("flaky_tests", 0.0)))
    flakiness_threshold = float(metrics.get("flakiness_threshold", 5.0))
    flakiness_prob = (
        flakiness / flakiness_threshold if flakiness_threshold > 0 else 0.0
    )

    recent_failure_rate = float(
        metrics.get("recent_failure_rate", metrics.get("failure_rate", 0.0))
    )

    instability = float(metrics.get("instability", 0.0))

    probs = [
        error_prob,
        flakiness_prob,
        recent_failure_rate,
        instability,
    ]
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    prob = max(probs)
    return max(0.0, min(1.0, prob))


@dataclass
class ScenarioScorecard:
    """Summary of scenario performance deltas."""

    scenario: str
    roi_delta: float
    raroi_delta: float
    metrics_delta: Dict[str, float]
    synergy_delta: Dict[str, float]
    recommendation: str | None = None
    status: str | None = None
    score: float | None = None


class ROITracker:
    """Monitor ROI change and determine when improvements diminish."""

    def __init__(
        self,
        window: int = 5,
        tolerance: float = 0.01,
        *,
        entropy_threshold: float | None = None,
        filter_outliers: bool = True,
        weights: Optional[List[float]] = None,
        resource_db: "ROIHistoryDB" | None = None,
        cluster_map: Optional[Dict[str, int]] = None,
        evaluation_window: int = 20,
        mae_threshold: float = 0.1,
        acc_threshold: float = 0.6,
        evaluate_every: int = 1,
        calibrate: bool | None = None,
        workflow_window: int = 20,
        confidence_threshold: float = 0.5,
        raroi_borderline_threshold: float = 0.0,
        borderline_bucket: BorderlineBucket | None = None,
        telemetry_backend: TelemetryBackend | None = None,
        telemetry_bootstrap_safe: bool | None = None,
        telemetry_timeout_s: float | None = None,
        results_db: "ROIResultsDB" | None = None,
    ) -> None:
        """Create a tracker for monitoring ROI deltas.

        Parameters
        ----------
        window:
            Number of recent deltas considered when evaluating convergence.
        tolerance:
            Threshold below which the average delta is considered negligible.
        entropy_threshold:
            Minimum ROI gain per unit entropy delta before entropy increases are
            treated as unproductive. Defaults to ``tolerance`` when omitted.
        filter_outliers:
            Whether to discard extreme deltas when updating the history.
        weights:
            Optional weights applied to the most recent ``window`` deltas when
            computing the stop criterion. Length must equal ``window`` and the
            values must sum to a non-zero amount.
        resource_db:
            Optional :class:`ResourcesBot.ROIHistoryDB` providing historical
            resource usage (CPU, memory, disk, time and GPU) to incorporate in
            forecasts.
        workflow_window:
            Size of the rolling window for per-workflow error metrics.
        confidence_threshold:
            Minimum confidence required to avoid flagging a workflow for review.
        raroi_borderline_threshold:
            RAROI value below which a workflow is considered borderline and
            therefore routed through the borderline bucket for micro-pilot
            evaluation.
        borderline_bucket:
            Optional :class:`BorderlineBucket` instance for tracking borderline
            workflows. A default instance is created when omitted.
        telemetry_backend:
            Optional backend used to persist prediction and drift telemetry.
            A default SQLite-backed implementation is used when not provided.
        telemetry_bootstrap_safe:
            When ``True`` the telemetry backend is initialised in bootstrap-safe
            mode, applying a short watchdog to SQLite configuration and
            falling back to in-memory/no-op telemetry on timeouts.
            Defaults to the value of ``BOOTSTRAP_WATCHDOG_ACTIVE`` when
            omitted.
        telemetry_timeout_s:
            Optional deadline applied to SQLite configuration when
            ``telemetry_bootstrap_safe`` is enabled.
        """
        self.roi_history: List[float] = []
        self.raroi_history: List[float] = []
        self.last_raroi: float | None = None
        self.last_confidence: float | None = None
        self.confidence_decay = 0.9
        self.last_predicted_roi: float | None = None
        self.confidence_history: List[float] = []
        self.mae_history: List[float] = []
        self.variance_history: List[float] = []
        self.final_roi_history: Dict[str, List[float]] = defaultdict(list)
        self.category_history: List[str] = []
        self.entropy_history: List[float] = []
        self.entropy_delta_history: List[float] = []
        self.needs_review: set[str] = set()
        self.confidence_threshold = float(confidence_threshold)
        self.raroi_borderline_threshold = float(raroi_borderline_threshold)
        self.borderline_bucket = borderline_bucket or BorderlineBucket()
        env_bootstrap = os.getenv(_BOOTSTRAP_WATCHDOG_ENV)
        telemetry_bootstrap_flag = (
            telemetry_bootstrap_safe
            if telemetry_bootstrap_safe is not None
            else env_bootstrap is not None
            and env_bootstrap.strip().lower() not in {"", "0", "false", "no", "off"}
        )
        self.telemetry = telemetry_backend or TelemetryBackend(
            _resolve_telemetry_db_path(),
            bootstrap_safe=telemetry_bootstrap_flag,
            init_timeout_s=telemetry_timeout_s,
        )
        self.results_db = results_db
        self.current_workflow_id: str | None = None
        self.current_run_id: str | None = None
        self.window = window
        self.tolerance = tolerance
        self.entropy_threshold = (
            tolerance if entropy_threshold is None else float(entropy_threshold)
        )
        self.filter_outliers = filter_outliers
        if weights is not None:
            if len(weights) != window:
                raise ValueError("len(weights) must equal window")
            arr = np.array(weights, dtype=float)
            if not arr.any():
                raise ValueError("weights must not sum to zero")
            self.weights = arr / arr.sum()
        else:
            self.weights = None
        self._poly = PolynomialFeatures(degree=2)
        self._model = LinearRegression()
        self.module_deltas: Dict[str, List[float]] = {}
        self.module_raroi: Dict[str, List[float]] = {}
        self.module_entropy_deltas: Dict[str, List[float]] = {}
        self.correlation_history: Dict[tuple[str, str], List[float]] = defaultdict(list)
        # Historical ROI deltas grouped by origin database.  The latest values
        # are exposed via :meth:`origin_db_deltas` for external consumers while
        # the full history remains available through
        # :attr:`origin_db_delta_history`.
        self._origin_db_deltas: Dict[str, List[float]] = {}
        self.db_roi_metrics: Dict[str, Dict[str, float]] = {}
        self.cluster_map: Dict[str, int] = dict(cluster_map or {})
        self.cluster_deltas: Dict[int, List[float]] = {}
        self.cluster_raroi: Dict[int, List[float]] = {}
        self.metrics_history: Dict[str, List[float]] = {
            "recovery_time": [],
            "latency_error_rate": [],
            "hostile_failures": [],
            "misuse_failures": [],
            "concurrency_throughput": [],
            "synergy_adaptability": [],
            "synergy_recovery_time": [],
            "synergy_discrepancy_count": [],
            "synergy_gpu_usage": [],
            "synergy_cpu_usage": [],
            "synergy_memory_usage": [],
            "synergy_long_term_lucrativity": [],
            "synergy_shannon_entropy": [],
            "synergy_flexibility": [],
            "synergy_energy_consumption": [],
            "synergy_profitability": [],
            "synergy_revenue": [],
            "synergy_projected_lucrativity": [],
            "synergy_maintainability": [],
            "synergy_code_quality": [],
            "synergy_network_latency": [],
            "synergy_throughput": [],
            "synergy_latency_error_rate": [],
            "synergy_hostile_failures": [],
            "synergy_misuse_failures": [],
            "synergy_concurrency_throughput": [],
            "synergy_risk_index": [],
            "synergy_safety_rating": [],
            "synergy_efficiency": [],
            "synergy_antifragility": [],
            "synergy_resilience": [],
            "synergy_security_score": [],
            "synergy_reliability": [],
            "roi_reliability": [],
            "long_term_roi_reliability": [],
        }
        self.synergy_metrics_history: Dict[str, List[float]] = {
            "synergy_adaptability": [],
            "synergy_recovery_time": [],
            "synergy_discrepancy_count": [],
            "synergy_gpu_usage": [],
            "synergy_cpu_usage": [],
            "synergy_memory_usage": [],
            "synergy_long_term_lucrativity": [],
            "synergy_shannon_entropy": [],
            "synergy_flexibility": [],
            "synergy_energy_consumption": [],
            "synergy_profitability": [],
            "synergy_revenue": [],
            "synergy_projected_lucrativity": [],
            "synergy_maintainability": [],
            "synergy_code_quality": [],
            "synergy_network_latency": [],
            "synergy_throughput": [],
            "synergy_latency_error_rate": [],
            "synergy_hostile_failures": [],
            "synergy_misuse_failures": [],
            "synergy_concurrency_throughput": [],
            "synergy_risk_index": [],
            "synergy_safety_rating": [],
            "synergy_efficiency": [],
            "synergy_antifragility": [],
            "synergy_resilience": [],
            "synergy_security_score": [],
            "synergy_roi_reliability": [],
            "synergy_reliability": [],
        }
        self.synergy_history: list[dict[str, float]] = []
        self.scenario_synergy: Dict[str, List[Dict[str, float]]] = {}
        self.scenario_roi_deltas: Dict[str, float] = {}
        self.scenario_raroi: Dict[str, float] = {}
        self.scenario_raroi_delta: Dict[str, float] = {}
        self.scenario_metrics_delta: Dict[str, Dict[str, float]] = {}
        self.scenario_synergy_delta: Dict[str, Dict[str, float]] = {}
        self._worst_scenario: str | None = None
        self._best_order: Optional[Tuple[int, int, int]] = None
        self.workflow_label: str | None = None
        self._order_history: Tuple[float, ...] = ()
        self.resource_db = resource_db
        self.resource_metrics: List[Tuple[float, float, float, float, float]] = []
        self.predicted_roi: List[float] = []
        self.actual_roi: List[float] = []
        # store per-workflow prediction and actual ROI histories
        self.workflow_window = int(workflow_window)
        # track per-workflow predicted and actual ROI values
        self.workflow_predicted_roi: Dict[str, List[float]] = defaultdict(list)
        self.workflow_actual_roi: Dict[str, List[float]] = defaultdict(list)
        self.workflow_confidence_history: Dict[str, List[float]] = defaultdict(list)
        self.workflow_mae_history: Dict[str, List[float]] = defaultdict(list)
        self.workflow_variance_history: Dict[str, List[float]] = defaultdict(list)
        # latest confidence score per workflow
        self.workflow_confidence_scores: Dict[str, float] = defaultdict(float)
        self.predicted_metrics: Dict[str, List[float]] = {}
        self.actual_metrics: Dict[str, List[float]] = {}
        self.predicted_classes: List[str] = []
        self.actual_classes: List[str] = []
        # track drift statistics for adaptive retraining
        self.drift_scores: List[float] = []
        self.drift_flags: List[bool] = []
        self.drift_metrics: Dict[str, float] = {}
        self.workflow_evaluation_metrics: Dict[str | None, Dict[str, float]] = {}
        # track horizon-wise errors and whether they compound over time
        self.horizon_mae_history: List[Dict[int, float]] = []
        self.compounding_flags: List[bool] = []
        self.evaluation_window = evaluation_window
        self.mae_threshold = mae_threshold
        self.acc_threshold = acc_threshold
        self.evaluate_every = max(1, int(evaluate_every))
        self._next_prediction: float | None = None
        self._next_category: str | None = None
        try:
            from .adaptive_roi_predictor import AdaptiveROIPredictor  # noqa: F401
        except Exception:
            pass
        self._adaptive_predictor = None
        if calibrate is None:
            calibrate = (
                os.getenv("ENABLE_TRUTH_CALIBRATION", "1").lower()
                not in {"0", "false"}
            )
        self.calibrate = calibrate
        self.truth_adapter: TruthAdapter | None = None
        if _TA_LOW_CONF_GAUGE is not None:
            _TA_LOW_CONF_GAUGE.set(0)
        if self.resource_db:
            try:
                df = self.resource_db.history()
                cols = [
                    c
                    for c in ("cpu", "memory", "disk", "time", "gpu")
                    if c in df.columns
                ]
                if cols:
                    data = df[cols].values.tolist()
                    for row in data:
                        vals = [float(x) for x in row]
                        if len(vals) < 5:
                            vals.extend([0.0] * (5 - len(vals)))
                        self.resource_metrics.append(tuple(vals[:5]))
            except Exception:
                logger.exception("resource history fetch failed")

        # restore persisted per-workflow prediction history if available
        if not os.getenv("PYTEST_CURRENT_TEST"):
            try:
                self.load_prediction_history()
            except Exception:
                logger.exception("failed to load prediction history")

    # ------------------------------------------------------------------
    def _ensure_truth_adapter(self) -> None:
        """Initialise the :class:`TruthAdapter` lazily when calibration is enabled."""

        if self.truth_adapter is not None or not self.calibrate:
            return

        try:
            self.truth_adapter = TruthAdapter(calibration_active=self.calibrate)
        except Exception:
            self.truth_adapter = None
            logger.exception("failed to initialise truth adapter")

    # ------------------------------------------------------------------
    @staticmethod
    def _ema(values: Iterable[float], alpha: float = 0.3) -> float:
        """Return exponential moving average for ``values``."""
        it = iter(values)
        try:
            ema = float(next(it))
        except StopIteration:
            return 0.0
        for val in it:
            ema = alpha * float(val) + (1.0 - alpha) * ema
        return float(ema)

    # ------------------------------------------------------------------
    def register_metrics(self, *names: str) -> None:
        """Ensure ``metrics_history`` contains ``names`` padded to current length."""

        for raw in names:
            name = str(raw)
            target = (
                self.synergy_metrics_history
                if name.startswith("synergy_")
                else self.metrics_history
            )
            if name not in target:
                target[name] = [0.0] * len(self.roi_history)
            self.predicted_metrics.setdefault(name, [])
            self.actual_metrics.setdefault(name, [])
            pair_map = {
                "safety_rating": "synergy_safety_rating",
                "security_score": "synergy_security_score",
                "discrepancy_count": "synergy_discrepancy_count",
                "gpu_usage": "synergy_gpu_usage",
                "cpu_usage": "synergy_cpu_usage",
                "memory_usage": "synergy_memory_usage",
                "long_term_lucrativity": "synergy_long_term_lucrativity",
                "shannon_entropy": "synergy_shannon_entropy",
                "flexibility": "synergy_flexibility",
                "energy_consumption": "synergy_energy_consumption",
                "profitability": "synergy_profitability",
                "revenue": "synergy_revenue",
                "projected_lucrativity": "synergy_projected_lucrativity",
                "maintainability": "synergy_maintainability",
                "code_quality": "synergy_code_quality",
                "network_latency": "synergy_network_latency",
                "throughput": "synergy_throughput",
                "latency_error_rate": "synergy_latency_error_rate",
                "hostile_failures": "synergy_hostile_failures",
                "misuse_failures": "synergy_misuse_failures",
                "concurrency_throughput": "synergy_concurrency_throughput",
                "risk_index": "synergy_risk_index",
                "recovery_time": "synergy_recovery_time",
                "reliability": "synergy_reliability",
                "adaptability": "synergy_adaptability",
                "efficiency": "synergy_efficiency",
                "antifragility": "synergy_antifragility",
                "resilience": "synergy_resilience",
            }
            syn_name = pair_map.get(name)
            if syn_name and syn_name not in self.synergy_metrics_history:
                self.synergy_metrics_history[syn_name] = [0.0] * len(self.roi_history)
                self.predicted_metrics.setdefault(syn_name, [])
                self.actual_metrics.setdefault(syn_name, [])

    # ------------------------------------------------------------------
    def merge_history(self, other: "ROITracker") -> None:
        """Merge ROI and metrics histories from ``other`` into this tracker."""

        pre_len = len(self.roi_history)
        self.roi_history.extend(other.roi_history)
        self.raroi_history.extend(getattr(other, "raroi_history", []))
        self.confidence_history.extend(other.confidence_history)
        self.mae_history.extend(getattr(other, "mae_history", []))
        self.variance_history.extend(getattr(other, "variance_history", []))
        for wf, hist in getattr(other, "final_roi_history", {}).items():
            self.final_roi_history.setdefault(wf, []).extend(hist)
        self.needs_review.update(getattr(other, "needs_review", set()))
        self.entropy_history.extend(getattr(other, "entropy_history", []))
        self.entropy_delta_history.extend(getattr(other, "entropy_delta_history", []))
        delta = len(other.roi_history)

        for name, hist in self.metrics_history.items():
            hist.extend(other.metrics_history.get(name, [0.0] * delta))
        for name, hist in other.metrics_history.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = [0.0] * pre_len + list(hist)

        for name, hist in self.synergy_metrics_history.items():
            hist.extend(other.synergy_metrics_history.get(name, [0.0] * delta))
        for name, hist in other.synergy_metrics_history.items():
            if name not in self.synergy_metrics_history:
                self.synergy_metrics_history[name] = [0.0] * pre_len + list(hist)

        self.synergy_history.extend(other.synergy_history)
        self.predicted_roi.extend(getattr(other, "predicted_roi", []))
        self.actual_roi.extend(getattr(other, "actual_roi", []))

        for name, hist in other.predicted_metrics.items():
            self.predicted_metrics.setdefault(name, []).extend(hist)
        for name, hist in other.actual_metrics.items():
            self.actual_metrics.setdefault(name, []).extend(hist)

        self.resource_metrics.extend(getattr(other, "resource_metrics", []))
        for name, deltas in getattr(other, "module_deltas", {}).items():
            self.module_deltas.setdefault(name, []).extend(deltas)
        for name, deltas in getattr(other, "module_raroi", {}).items():
            self.module_raroi.setdefault(name, []).extend(deltas)
        for name, deltas in getattr(other, "module_entropy_deltas", {}).items():
            self.module_entropy_deltas.setdefault(name, []).extend(deltas)
        for name, deltas in getattr(other, "cluster_deltas", {}).items():
            self.cluster_deltas.setdefault(name, []).extend(deltas)
        for name, deltas in getattr(other, "cluster_raroi", {}).items():
            self.cluster_raroi.setdefault(name, []).extend(deltas)
        for db, deltas in getattr(other, "origin_db_delta_history", {}).items():
            self._origin_db_deltas.setdefault(db, []).extend(deltas)

    # ------------------------------------------------------------------
    def diminishing(self) -> float:
        """Return the ROI delta threshold considered negligible.

        The base ``tolerance`` is scaled by the volatility of recent ROI
        deltas.  The standard deviation of the last ``window`` entries of
        :attr:`roi_history` is computed and used to adjust the threshold so
        that noisier histories result in a larger threshold.
        """

        history = self.roi_history[-self.window :] if self.roi_history else []
        if history:
            stddev = float(np.std(history))
        else:
            stddev = 0.0
        return float(self.tolerance * (1.0 + stddev))

    # ------------------------------------------------------------------
    def _regression(self) -> Tuple[Optional[int], List[float]]:
        """Return vertex index and predictions from quadratic regression."""
        if len(self.roi_history) < 3:
            return None, []
        x = np.arange(len(self.roi_history)).reshape(-1, 1)
        y = np.array(self.roi_history)
        x_poly = self._poly.fit_transform(x)
        self._model.fit(x_poly, y)
        preds = self._model.predict(x_poly).tolist()
        a = float(self._model.coef_[2])
        b = float(self._model.coef_[1])
        if abs(a) < 1e-9:
            vertex = None
        else:
            vertex = int(round(-b / (2 * a)))
        return vertex, preds

    # ------------------------------------------------------------------
    def set_run_context(
        self,
        workflow_id: str,
        run_id: str,
        results_db: "ROIResultsDB" | None = None,
    ) -> None:
        """Configure persistence context for ROI deltas.

        When set, subsequent :meth:`update` calls persist per-module ROI deltas
        to :class:`ROIResultsDB`.  Passing ``results_db`` overrides the tracker
        instance provided during initialisation, allowing callers to inject a
        database on demand.
        """

        self.current_workflow_id = workflow_id
        self.current_run_id = run_id
        if results_db is not None:
            self.results_db = results_db

    # ------------------------------------------------------------------
    def _filter_value(self, delta: float) -> Optional[float]:
        """Return ``delta`` unless it's an outlier based on IQR or z-score."""
        if not self.filter_outliers:
            return delta
        if len(self.roi_history) < 4:
            return delta
        data = np.array(self.roi_history, dtype=float)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        if delta < lower or delta > upper:
            return None
        return delta

    # ------------------------------------------------------------------
    def record_prediction(
        self,
        predicted: float | Sequence[float],
        actual: float | Sequence[float],
        predicted_class: str | None = None,
        actual_class: str | None = None,
        confidence: float | None = None,
        workflow_id: str | None = None,
        modules: Optional[List[str]] = None,
        final_score: float | None = None,
    ) -> None:
        """Store prediction details and log prediction stats.

        ``predicted`` and ``actual`` may contain multiple horizon values.  The
        first element is used for the standard scalar history while the full
        sequences are persisted for horizon-specific evaluation.
        """

        pred_seq = (
            [float(x) for x in predicted]
            if isinstance(predicted, (list, tuple, np.ndarray))
            else [float(predicted)]
        )
        act_seq = (
            [float(x) for x in actual]
            if isinstance(actual, (list, tuple, np.ndarray))
            else [float(actual)]
        )
        low_conf = False
        if self.calibrate:
            self._ensure_truth_adapter()
        if self.truth_adapter is not None:
            try:
                arr = np.array(pred_seq, dtype=float).reshape(-1, 1)
                realish, low_conf = self.truth_adapter.predict(arr)
                pred_seq = [float(x) for x in realish]
                if _TA_LOW_CONF_GAUGE is not None:
                    _TA_LOW_CONF_GAUGE.set(1 if low_conf else 0)
                if low_conf and self._adaptive_predictor is not None:
                    logger.warning(
                        "truth adapter low confidence; consider retraining",
                        extra=log_record(low_confidence=True),
                    )
                    try:
                        self._adaptive_predictor.train()
                    except Exception:
                        logger.exception("failed to trigger adaptive ROI retrain")
            except Exception:
                logger.exception("truth adapter predict failed")

        # update primary workflow and global history before drift checks
        pred_val = float(pred_seq[0]) if pred_seq else None
        act_val = float(act_seq[0]) if act_seq else None
        if pred_val is not None:
            self.predicted_roi.append(pred_val)
            self.last_predicted_roi = pred_val
            if workflow_id is not None:
                wf = str(workflow_id)
                self.workflow_predicted_roi[wf].append(pred_val)
                self.workflow_predicted_roi[wf] = self.workflow_predicted_roi[wf][
                    -self.workflow_window:
                ]
        if act_val is not None:
            self.actual_roi.append(act_val)
            if workflow_id is not None:
                wf = str(workflow_id)
                self.workflow_actual_roi[wf].append(act_val)
                self.workflow_actual_roi[wf] = self.workflow_actual_roi[wf][
                    -self.workflow_window:
                ]
        workflows: List[str] = []
        if modules:
            workflows.extend(str(m) for m in modules)
        if workflow_id is not None:
            workflows.append(str(workflow_id))
        for wf in workflows:
            if wf != (None if workflow_id is None else str(workflow_id)):
                self.workflow_predicted_roi[wf].append(float(pred_seq[0]))
                self.workflow_actual_roi[wf].append(float(act_seq[0]))
            # trim to rolling window
            self.workflow_predicted_roi[wf] = self.workflow_predicted_roi[wf][-self.workflow_window:]
            self.workflow_actual_roi[wf] = self.workflow_actual_roi[wf][-self.workflow_window:]
            conf = self.workflow_confidence(wf, self.workflow_window)
            self.workflow_confidence_history[wf].append(conf)
            mae_val = self.workflow_mae(wf, self.workflow_window)
            var_val = self.workflow_variance(wf, self.workflow_window)
            self.workflow_mae_history[wf].append(mae_val)
            self.workflow_variance_history[wf].append(var_val)
            if len(self.workflow_confidence_history[wf]) > self.workflow_window:
                self.workflow_confidence_history[wf] = self.workflow_confidence_history[wf][-self.workflow_window:]
            if len(self.workflow_mae_history[wf]) > self.workflow_window:
                self.workflow_mae_history[wf] = self.workflow_mae_history[wf][-self.workflow_window:]
            if len(self.workflow_variance_history[wf]) > self.workflow_window:
                self.workflow_variance_history[wf] = self.workflow_variance_history[wf][-self.workflow_window:]
            needs_review = conf < self.confidence_threshold
            logger.info(
                "workflow prediction metrics",
                extra=log_record(
                    workflow=wf,
                    predicted=float(pred_seq[0]),
                    actual=float(act_seq[0]),
                    workflow_mae=mae_val,
                    workflow_variance=var_val,
                    workflow_confidence=conf,
                    human_review=needs_review if needs_review else None,
                ),
            )
            if needs_review:
                logger.info(
                    "workflow flagged for human review",
                    extra=log_record(
                        workflow=wf,
                        workflow_confidence=conf,
                        threshold=self.confidence_threshold,
                        human_review=True,
                    ),
                )
            try:
                if _me is not None:
                    _me.workflow_mae.labels(workflow=wf).set(mae_val)
                    _me.workflow_variance.labels(workflow=wf).set(var_val)
                    _me.workflow_confidence.labels(workflow=wf).set(conf)
            except Exception:
                pass
        try:
            ent_hist = self.metrics_history.get("synergy_shannon_entropy", [])
            if len(ent_hist) >= 2:
                entropy_delta = float(ent_hist[-1]) - float(ent_hist[-2])
                from .code_database import PatchHistoryDB

                patch_db = PatchHistoryDB()
                with patch_db._connect() as conn:
                    row = conn.execute(
                        "SELECT filename, complexity_delta FROM patch_history ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                if row:
                    mod_name, complexity_delta = row
                    complexity_delta = float(complexity_delta or 0.0)
                    if complexity_delta:
                        ratio = entropy_delta / complexity_delta
                        self.module_entropy_deltas.setdefault(str(mod_name), []).append(
                            ratio
                        )
        except Exception:
            pass

        if predicted_class is None:
            predicted_class = self._next_category
        if actual_class is None and self._adaptive_predictor is not None:
            try:
                feats = [[float(x)] for x in (self.roi_history + [act_seq[0]])] or [[0.0]]
                try:
                    _, actual_class, _, _ = self._adaptive_predictor.predict(
                        feats, horizon=len(feats)
                    )
                except TypeError:
                    _, actual_class, _, _ = self._adaptive_predictor.predict(feats)
            except Exception:
                actual_class = None
        if predicted_class is not None and actual_class is not None:
            self.record_class_prediction(predicted_class, actual_class)
        wf_mae = (
            self.workflow_mae(workflow_id)
            if workflow_id is not None
            else self.rolling_mae()
        )
        wf_var = (
            self.workflow_variance(workflow_id)
            if workflow_id is not None
            else float(np.var(self.actual_roi)) if self.actual_roi else 0.0
        )
        conf_val = confidence if confidence is not None else (
            self.workflow_confidence(workflow_id) if workflow_id is not None else 0.0
        )

        logger.info(
            "roi prediction",
            extra=log_record(
                iteration=len(self.predicted_roi),
                predicted=float(pred_seq[0]),
                actual=float(act_seq[0]),
                predicted_class=predicted_class,
                actual_class=actual_class,
                workflow_confidence=conf_val,
                workflow_id=workflow_id,
                workflow_mae=wf_mae,
                workflow_variance=wf_var,
            ),
        )
        try:
            from . import metrics_exporter as _me
            for g in (
                _me.prediction_error,
                _me.prediction_mae,
                _me.prediction_reliability,
            ):
                for attr in ("_metrics", "_values"):
                    try:
                        getattr(g, attr).clear()
                    except Exception:
                        continue
            err = abs(float(pred_seq[0]) - float(act_seq[0]))
            _me.prediction_error.labels(metric="roi").set(err)
            _me.prediction_mae.labels(metric="roi").set(self.rolling_mae())
            _me.prediction_reliability.labels(metric="roi").set(
                self.reliability()
            )
            if workflow_id is not None:
                try:
                    _me.workflow_mae.labels(workflow=workflow_id).set(wf_mae)
                    _me.workflow_variance.labels(workflow=workflow_id).set(wf_var)
                    _me.workflow_confidence.labels(workflow=workflow_id).set(conf_val)
                except Exception:
                    pass
        except Exception:
            pass

        drift = self.check_prediction_drift()
        self.drift_metrics["mae"] = self.rolling_mae()
        self.drift_metrics["accuracy"] = self.classification_accuracy()
        readiness = self._compute_readiness()
        if self.telemetry is not None:
            try:
                self.telemetry.log_prediction(
                    None if workflow_id is None else str(workflow_id),
                    float(pred_seq[0]) if pred_seq else None,
                    float(act_seq[0]) if act_seq else None,
                    self.last_confidence,
                    dict(self.scenario_roi_deltas),
                    drift,
                    readiness,
                )
            except Exception:
                logger.exception("failed to log telemetry event")

        # persist prediction event with additional telemetry
        scen_id: str | None = None
        scen_delta: float | None = None
        if self.scenario_roi_deltas:
            scen_id, scen_delta = self.biggest_drop()
        drift_score = self.drift_scores[-1] if self.drift_scores else None
        self.record_roi_prediction(
            pred_seq,
            act_seq,
            predicted_class,
            actual_class,
            confidence if confidence is not None else self.last_confidence,
            workflow_id,
            final_score=final_score,
            scenario_id=scen_id,
            scenario_roi_delta=scen_delta,
            drift_score=drift_score,
            drift_flag=drift,
            skip_history=True,
        )

    def record_roi_prediction(
        self,
        predicted: Sequence[float],
        actual: Sequence[float],
        predicted_class: str | None = None,
        actual_class: str | None = None,
        confidence: float | None = None,
        workflow_id: str | None = None,
        final_score: float | None = None,
        *,
        scenario_id: str | None = None,
        scenario_roi_delta: float | None = None,
        drift_score: float | None = None,
        drift_flag: bool | None = None,
        skip_history: bool = False,
    ) -> None:
        """Persist an ROI prediction event to ``roi_events.db``."""

        if not skip_history:
            pred_val = float(predicted[0]) if predicted else None
            act_val = float(actual[0]) if actual else None
            if pred_val is not None:
                self.predicted_roi.append(pred_val)
                self.last_predicted_roi = pred_val
                if workflow_id is not None:
                    wf = str(workflow_id)
                    self.workflow_predicted_roi[wf].append(pred_val)
                    self.workflow_predicted_roi[wf] = self.workflow_predicted_roi[wf][
                        -self.workflow_window:
                    ]
            if act_val is not None:
                self.actual_roi.append(act_val)
                if workflow_id is not None:
                    wf = str(workflow_id)
                    self.workflow_actual_roi[wf].append(act_val)
                    self.workflow_actual_roi[wf] = self.workflow_actual_roi[wf][
                        -self.workflow_window:
                    ]
        try:
            deltas = (
                {str(scenario_id): float(scenario_roi_delta)}
                if scenario_id is not None and scenario_roi_delta is not None
                else {}
            )
            tb.log_roi_event(
                predicted_roi=float(predicted[0]) if predicted else None,
                actual_roi=float(actual[0]) if actual else None,
                confidence=confidence,
                scenario_deltas=deltas,
                drift_flag=bool(drift_flag) if drift_flag is not None else False,
                workflow_id=workflow_id,
                predicted_class=predicted_class,
                actual_class=actual_class,
                predicted_horizons=[float(x) for x in predicted],
                actual_horizons=[float(x) for x in actual],
                predicted_categories=[predicted_class] if predicted_class is not None else [],
                actual_categories=[actual_class] if actual_class is not None else [],
            )
        except Exception:
            logger.exception("failed to log roi prediction event")

    def record_class_prediction(self, predicted: str, actual: str) -> None:
        """Store predicted and actual ROI classes."""
        self.predicted_classes.append(str(predicted))
        self.actual_classes.append(str(actual))

    def load_prediction_history(self, path: str = "roi_events.db") -> None:
        """Load prior prediction events to rebuild workflow histories."""
        if not os.path.exists(path):
            return
        try:
            conn = router.get_connection("roi_prediction_events")
            try:
                cur = conn.execute(
                    "SELECT workflow, predicted_roi, actual_roi, confidence FROM roi_prediction_events ORDER BY ts"
                )
            except sqlite3.OperationalError:
                cur = conn.execute(
                    "SELECT workflow_id, predicted_roi, actual_roi, confidence FROM roi_prediction_events ORDER BY ts"
                )
            rows = cur.fetchall()
        except Exception:
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for wf, pred, act, conf in rows:
            wf_id = None if wf is None else str(wf)
            if pred is not None:
                val = float(pred)
                self.predicted_roi.append(val)
                if wf_id is not None:
                    self.workflow_predicted_roi[wf_id].append(val)
                    self.workflow_predicted_roi[wf_id] = self.workflow_predicted_roi[wf_id][
                        -self.workflow_window:
                    ]
            if act is not None:
                val = float(act)
                self.actual_roi.append(val)
                if wf_id is not None:
                    self.workflow_actual_roi[wf_id].append(val)
                    self.workflow_actual_roi[wf_id] = self.workflow_actual_roi[wf_id][
                        -self.workflow_window:
                    ]
            if conf is not None and wf_id is not None:
                self.workflow_confidence_history[wf_id].append(float(conf))

    # ------------------------------------------------------------------
    def fetch_prediction_events(
        self,
        workflow_id: str | None = None,
        *,
        start_ts: str | None = None,
        end_ts: str | None = None,
        limit: int | None = None,
        path: str = "roi_events.db",
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return persisted prediction events with telemetry details.

        ``source_menace_id`` overrides the current menace identifier when
        querying shared databases.
        """

        conn = router.get_connection("roi_prediction_events")
        try:
            cur = conn.cursor()
            base = (
                "SELECT workflow_id, predicted_roi, actual_roi, confidence, "
                "scenario_deltas, drift_flag, ts FROM roi_prediction_events"
            )
            menace_id = source_menace_id or router.menace_id
            clause, scope_params = build_scope_clause(
                "roi_prediction_events", scope, menace_id
            )
            base = apply_scope(base, clause)
            params: List[Any] = [*scope_params]
            if workflow_id is not None:
                base = apply_scope(base, "workflow_id = ?")
                params.append(workflow_id)
            if start_ts is not None:
                base = apply_scope(base, "ts >= ?")
                params.append(start_ts)
            if end_ts is not None:
                base = apply_scope(base, "ts <= ?")
                params.append(end_ts)
            base += " ORDER BY ts"
            if limit is not None:
                base += " LIMIT ?"
                params.append(int(limit))
            cur.execute(base, params)
            rows = cur.fetchall()
            events: List[Dict[str, Any]] = []
            for wf, pred, act, conf, deltas, flag, ts in rows:
                events.append(
                    {
                        "workflow_id": wf,
                        "predicted_roi": pred,
                        "actual_roi": act,
                        "confidence": conf,
                        "scenario_deltas": json.loads(deltas) if deltas else {},
                        "drift_flag": bool(flag) if flag is not None else None,
                        "ts": ts,
                    }
                )
            return events
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def fetch_drift_history(
        self,
        workflow_id: str | None = None,
        *,
        limit: int | None = None,
        path: str = "roi_events.db",
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return drift scores and flags for dashboards.

        ``source_menace_id`` overrides the current menace identifier when
        querying shared databases.
        """

        conn = router.get_connection("roi_prediction_events")
        try:
            cur = conn.cursor()
            base = "SELECT drift_flag, ts FROM roi_prediction_events"
            menace_id = source_menace_id or router.menace_id
            clause, scope_params = build_scope_clause(
                "roi_prediction_events", scope, menace_id
            )
            base = apply_scope(base, clause)
            params: List[Any] = [*scope_params]
            if workflow_id is not None:
                base = apply_scope(base, "workflow_id = ?")
                params.append(workflow_id)
            base += " ORDER BY ts"
            if limit is not None:
                base += " LIMIT ?"
                params.append(int(limit))
            cur.execute(base, params)
            rows = cur.fetchall()
            return [
                {"drift_flag": bool(f), "ts": ts}
                for f, ts in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def roi_event_history(
        self,
        workflow_id: str | None = None,
        *,
        limit: int | None = None,
        path: str = "roi_events.db",
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Wrapper around :func:`telemetry_backend.fetch_roi_events`.

        ``source_menace_id`` forwards to the underlying fetch helper.
        """

        return tb.fetch_roi_events(
            workflow_id=workflow_id,
            limit=limit,
            db_path=path,
            scope=scope,
            source_menace_id=source_menace_id,
        )

    # ------------------------------------------------------------------
    def evaluate_model(
        self,
        window: int = 20,
        mae_threshold: float = 0.1,
        acc_threshold: float = 0.6,
        history_path: str | None = None,
        roi_events_path: str | None = None,
        drift_threshold: float = 0.3,
    ) -> tuple[float, float]:
        """Evaluate prediction accuracy and trigger retraining when needed.

        Prediction performance is calculated from persisted events stored in
        ``roi_events.db``. Both ``history_path`` and ``roi_events_path`` are
        resolved relative to the project root via :func:`resolve_path`. When
        recent mean absolute error or classification accuracy fall outside the
        provided thresholds the adaptive ROI model is retrained using
        :class:`AdaptiveROIPredictor`.
        """

        if history_path is None:
            try:
                history_path = str(resolve_path("sandbox_data/roi_history.json"))
            except FileNotFoundError:
                history_path = "sandbox_data/roi_history.json"
        else:
            try:
                history_path = str(resolve_path(history_path))
            except FileNotFoundError:
                history_path = str(Path(history_path))
        if roi_events_path is None:
            try:
                roi_events_path = str(resolve_path("roi_events.db"))
            except FileNotFoundError:
                roi_events_path = "roi_events.db"
        else:
            try:
                roi_events_path = str(resolve_path(roi_events_path))
            except FileNotFoundError:
                roi_events_path = str(Path(roi_events_path))
        logger.debug(
            "evaluate_model using history_path=%s, roi_events_path=%s",
            history_path,
            roi_events_path,
        )

        try:
            conn = router.get_connection("roi_prediction_events")
            try:
                cur = conn.execute(
                    "SELECT workflow_id, predicted_roi, actual_roi, predicted_class, actual_class, predicted_horizons, actual_horizons, predicted_categories, actual_categories "
                    "FROM roi_prediction_events ORDER BY ts DESC LIMIT ?",
                    (int(window),),
                )
            except sqlite3.OperationalError:
                try:
                    cur = conn.execute(
                        "SELECT workflow_id, predicted_roi, actual_roi, predicted_class, actual_class, predicted_horizons, actual_horizons "
                        "FROM roi_prediction_events ORDER BY ts DESC LIMIT ?",
                        (int(window),),
                    )
                except sqlite3.OperationalError:
                    cur = conn.execute(
                        "SELECT workflow_id, predicted_roi, actual_roi, predicted_class, actual_class "
                        "FROM roi_prediction_events ORDER BY ts DESC LIMIT ?",
                        (int(window),),
                    )
            rows = cur.fetchall()
        except Exception:
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass

        wf_groups: Dict[str | None, List[tuple]] = defaultdict(list)
        for r in rows:
            wf_groups[r[0]].append(r[1:])

        preds: List[float] = []
        acts: List[float] = []
        errors: List[float] = []
        cls_pairs_all: List[tuple] = []
        horizon_errs: dict[int, list[float]] = defaultdict(list)
        self.workflow_evaluation_metrics = {}
        for wf_id, wf_rows in wf_groups.items():
            wf_preds = [float(x[0]) for x in wf_rows]
            wf_acts = [float(x[1]) for x in wf_rows[: len(wf_preds)]]
            wf_errors = [abs(p - a) for p, a in zip(wf_preds, wf_acts)]
            wf_mae = float(np.mean(wf_errors)) if wf_errors else 0.0
            wf_cls_pairs = [
                (x[2], x[3])
                for x in wf_rows
                if x[2] is not None and x[3] is not None
            ]
            if wf_cls_pairs:
                pc, ac = zip(*wf_cls_pairs)
                wf_acc = float((np.asarray(pc) == np.asarray(ac)).mean())
            else:
                wf_acc = 0.0
            self.workflow_evaluation_metrics[wf_id] = {
                "mae": wf_mae,
                "accuracy": wf_acc,
            }
            preds.extend(wf_preds)
            acts.extend(wf_acts)
            errors.extend(wf_errors)
            cls_pairs_all.extend(wf_cls_pairs)
            for x in wf_rows:
                if len(x) >= 6:
                    try:
                        ph = json.loads(x[4] or "[]")
                        ah = json.loads(x[5] or "[]")
                    except Exception:
                        ph, ah = [], []
                    for i, (p, a) in enumerate(zip(ph, ah), start=1):
                        horizon_errs[i].append(abs(float(p) - float(a)))

        mae = float(np.mean(errors)) if errors else 0.0
        mae_by_horizon = {
            h: float(np.mean(v)) for h, v in horizon_errs.items() if v
        }
        self.horizon_mae_history.append(mae_by_horizon)
        compounding = False
        if mae_by_horizon:
            ordered = [mae_by_horizon[h] for h in sorted(mae_by_horizon)]
            for prev, curr in zip(ordered, ordered[1:]):
                if curr > prev:
                    compounding = True
                    break
        self.compounding_flags.append(compounding)

        def detect_drift(errs: list[float]) -> float:
            """Return magnitude of error drift between recent and older windows."""
            if len(errs) < 4:
                return 0.0
            split = max(1, len(errs) // 2)
            recent = np.mean(errs[:split])
            history = np.mean(errs[split:]) if errs[split:] else 0.0
            return float(abs(recent - history))

        drift_score = detect_drift(errors)
        drift_detected = drift_score > float(drift_threshold)
        self.drift_scores.append(drift_score)
        self.drift_flags.append(drift_detected)

        if cls_pairs_all:
            pc, ac = zip(*cls_pairs_all)
            acc = float((np.asarray(pc) == np.asarray(ac)).mean())
        else:
            acc = 0.0

        self.drift_metrics["mae"] = mae
        self.drift_metrics["accuracy"] = acc
        if self._adaptive_predictor is None:
            try:
                from .adaptive_roi_predictor import AdaptiveROIPredictor

                self._adaptive_predictor = AdaptiveROIPredictor()
            except Exception:
                self._adaptive_predictor = None
        if self._adaptive_predictor is not None:
            try:
                if hasattr(self._adaptive_predictor, "record_drift"):
                    self._adaptive_predictor.record_drift(
                        acc,
                        mae,
                        acc_threshold=acc_threshold,
                        mae_threshold=mae_threshold,
                        retrain=True,
                    )
                elif hasattr(self._adaptive_predictor, "train"):
                    self._adaptive_predictor.train()
            except Exception:
                logger.exception("failed to update adaptive ROI predictor")

        return acc, mae

    # ------------------------------------------------------------------
    def check_prediction_drift(self) -> bool:
        """Flag prediction drift based on rolling MAE and retrain when needed."""

        if len(self.predicted_roi) < self.evaluation_window:
            return False
        mae = self.rolling_mae(self.evaluation_window)
        drift = mae > float(self.mae_threshold)
        self.drift_scores.append(mae)
        self.drift_flags.append(drift)
        if drift:
            try:
                if self._adaptive_predictor is None:
                    from .adaptive_roi_predictor import AdaptiveROIPredictor

                    self._adaptive_predictor = AdaptiveROIPredictor()
                self._adaptive_predictor.train()
            except Exception:
                logger.exception("failed to trigger adaptive ROI retrain")
        if drift:
            current = self.last_confidence if self.last_confidence is not None else 1.0
            self.last_confidence = current * self.confidence_decay
        readiness = self._compute_readiness()
        if self.telemetry is not None:
            try:
                pred = self.predicted_roi[-1] if self.predicted_roi else None
                act = self.actual_roi[-1] if self.actual_roi else None
                self.telemetry.log_prediction(
                    None,
                    pred,
                    act,
                    self.last_confidence,
                    None,
                    drift,
                    readiness,
                )
            except Exception:
                logger.exception("failed to log drift telemetry")
        return drift

    def _compute_readiness(self) -> float:
        """Return readiness score from latest metrics."""

        reliability = float(self.reliability())
        safety_hist = (
            self.metrics_history.get("security_score")
            or self.metrics_history.get("synergy_safety_rating")
            or [1.0]
        )
        safety = float(safety_hist[-1]) if safety_hist else 1.0
        res_hist = (
            self.metrics_history.get("synergy_resilience")
            or self.metrics_history.get("resilience")
            or [1.0]
        )
        resilience = float(res_hist[-1]) if res_hist else 1.0
        raroi = float(self.last_raroi or 0.0)
        return compute_readiness(raroi, reliability, safety, resilience)

    def record_metric_prediction(
        self, metric: str, predicted: float, actual: float
    ) -> None:
        """Store ``predicted`` and ``actual`` values for ``metric``."""
        name = str(metric)
        self.predicted_metrics.setdefault(name, []).append(float(predicted))
        self.actual_metrics.setdefault(name, []).append(float(actual))
        try:
            from . import metrics_exporter as _me
            for g in (
                _me.prediction_error,
                _me.prediction_mae,
                _me.prediction_reliability,
            ):
                for attr in ("_metrics", "_values"):
                    try:
                        getattr(g, attr).clear()
                    except Exception:
                        continue
            err = abs(float(predicted) - float(actual))
            _me.prediction_error.labels(metric=name).set(err)
            _me.prediction_mae.labels(metric=name).set(
                self.rolling_mae_metric(name)
            )
            _me.prediction_reliability.labels(metric=name).set(
                self.reliability(metric=name)
            )
        except Exception:
            pass

    @property
    def predicted_long_term_roi(self) -> List[float]:
        """Recorded long-term ROI predictions."""
        return self.predicted_metrics.get("long_term_roi", [])

    @property
    def actual_long_term_roi(self) -> List[float]:
        """Recorded long-term ROI outcomes."""
        return self.actual_metrics.get("long_term_roi", [])

    def record_long_term_roi(self, predicted: float, actual: float) -> None:
        """Store predicted and actual long-term ROI values."""
        self.record_metric_prediction("long_term_roi", predicted, actual)

    # ------------------------------------------------------------------
    def rolling_mae(self, window: int | None = None) -> float:
        """Return mean absolute error for the last ``window`` predictions."""
        if not self.predicted_roi:
            return 0.0
        preds = self.predicted_roi[-window:] if window else self.predicted_roi
        acts = self.actual_roi[-len(preds) :]
        arr = np.abs(np.array(preds) - np.array(acts))
        return float(arr.mean()) if arr.size else 0.0

    def rolling_mae_metric(self, metric: str, window: int | None = None) -> float:
        """Return MAE for ``metric`` predictions over ``window`` samples."""
        preds = self.predicted_metrics.get(metric, [])
        if not preds:
            return 0.0
        preds = preds[-window:] if window else preds
        acts = self.actual_metrics.get(metric, [])[-len(preds) :]
        arr = np.abs(np.array(preds) - np.array(acts))
        return float(arr.mean()) if arr.size else 0.0

    # ------------------------------------------------------------------
    def category_summary(self) -> Dict[str, int]:
        """Return counts of predicted ROI categories."""

        return dict(Counter(self.category_history))

    # ------------------------------------------------------------------
    def classification_accuracy(self, window: int | None = None) -> float:
        """Return accuracy of ROI class predictions.

        Parameters
        ----------
        window:
            Optional number of recent samples to evaluate. When omitted the
            entire history is used.
        """

        if not self.predicted_classes:
            return 0.0
        pc = self.predicted_classes[-window:] if window else self.predicted_classes
        ac = self.actual_classes[-len(pc) :]
        if not ac:
            return 0.0
        return float((np.asarray(pc) == np.asarray(ac)).mean())

    # ------------------------------------------------------------------
    def class_counts(self, window: int | None = None) -> Dict[str, Dict[str, int]]:
        """Return distribution of predicted and actual ROI classes."""

        preds = self.predicted_classes[-window:] if window else self.predicted_classes
        acts = self.actual_classes[-window:] if window else self.actual_classes
        return {"predicted": dict(Counter(preds)), "actual": dict(Counter(acts))}

    # ------------------------------------------------------------------
    def confusion_matrix(self, window: int | None = None) -> Dict[str, Dict[str, int]]:
        """Return confusion matrix for predicted vs actual ROI classes."""

        preds = self.predicted_classes[-window:] if window else self.predicted_classes
        acts = self.actual_classes[-len(preds) :]
        matrix: Dict[str, Dict[str, int]] = {}
        for a, p in zip(acts, preds):
            matrix.setdefault(a, {}).setdefault(p, 0)
            matrix[a][p] += 1
        return matrix

    # ------------------------------------------------------------------
    def rolling_accuracy_trend(self, window: int | None = None) -> List[float]:
        """Return rolling accuracy trend over history or ``window`` samples."""

        preds = self.predicted_classes
        acts = self.actual_classes
        if not preds or not acts:
            return []
        w = window or len(preds)
        trend: List[float] = []
        for i in range(1, len(preds) + 1):
            start = max(0, i - w)
            pc = preds[start:i]
            ac = acts[start:i]
            trend.append(float((np.asarray(pc) == np.asarray(ac)).mean()) if ac else 0.0)
        return trend[-w:]

    # ------------------------------------------------------------------
    def rolling_mae_trend(self, window: int | None = None) -> List[float]:
        """Return rolling MAE trend for ROI predictions."""

        preds = self.predicted_roi
        acts = self.actual_roi
        if not preds:
            return []
        w = window or len(preds)
        trend: List[float] = []
        for i in range(1, len(preds) + 1):
            start = max(0, i - w)
            p_slice = preds[start:i]
            a_slice = acts[start:i]
            arr = np.abs(np.array(p_slice) - np.array(a_slice))
            trend.append(float(arr.mean()) if arr.size else 0.0)
        return trend[-w:]

    # ------------------------------------------------------------------
    def workflow_mae(self, workflow_id: str, window: int | None = None) -> float:
        """Return mean absolute error for ``workflow_id`` predictions."""

        preds = self.workflow_predicted_roi.get(workflow_id, [])
        acts = self.workflow_actual_roi.get(workflow_id, [])
        if not preds or not acts:
            return 0.0
        if window is None:
            window = self.workflow_window
        if window > 0:
            preds = preds[-window:]
            acts = acts[-window:]
        arr = np.abs(np.array(preds) - np.array(acts))
        return float(arr.mean()) if arr.size else 0.0

    # ------------------------------------------------------------------
    def workflow_variance(self, workflow_id: str, window: int | None = None) -> float:
        """Return variance of actual ROI for ``workflow_id``."""

        acts = self.workflow_actual_roi.get(workflow_id, [])
        if not acts:
            return 0.0
        if window is None:
            window = self.workflow_window
        if window > 0:
            acts = acts[-window:]
        return float(np.var(acts)) if acts else 0.0

    # ------------------------------------------------------------------
    def workflow_confidence(self, workflow_id: str, window: int | None = None) -> float:
        """Return a confidence score for ``workflow_id`` predictions.

        The score is based on the mean absolute error and variance of the
        workflow's recent ROI predictions and is normalised to the ``[0, 1]``
        range using ``1 / (1 + mae + variance)``.
        """
        stored = self.workflow_confidence_scores.get(workflow_id)
        if stored:
            return max(0.0, min(1.0, float(stored)))
        mae = self.workflow_mae(workflow_id, window)
        variance = self.workflow_variance(workflow_id, window)
        confidence = 1.0 / (1.0 + mae + variance)
        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    def score_workflow(
        self, workflow_id: str, raroi: float, tau: float | None = None
    ) -> Tuple[float, bool, float]:
        """Return ``(final_score, needs_review, confidence)`` for ``workflow_id``.

        The final score scales ``raroi`` by the workflow's confidence. When the
        confidence falls below ``tau`` (or ``self.confidence_threshold`` when
        ``tau`` is ``None``) the workflow is flagged for manual review and
        automatic demotion is bypassed.
        """

        wf = str(workflow_id)
        stored_conf = self.workflow_confidence_scores.get(wf)
        if stored_conf:
            conf = max(0.0, min(1.0, float(stored_conf)))
        else:
            conf = self.workflow_confidence(wf, self.workflow_window)
        self.workflow_confidence_history[wf].append(conf)
        if len(self.workflow_confidence_history[wf]) > self.workflow_window:
            self.workflow_confidence_history[wf] = self.workflow_confidence_history[wf][
                -self.workflow_window :
            ]
        final_score = float(raroi) * conf
        self.final_roi_history.setdefault(wf, []).append(final_score)
        record_run(wf, final_score)
        threshold = self.confidence_threshold if tau is None else float(tau)
        needs_review = conf < threshold
        self.last_confidence = conf
        if raroi < self.raroi_borderline_threshold or needs_review:
            try:
                if self.borderline_bucket.get_candidate(wf) is None:
                    self.borderline_bucket.add_candidate(wf, float(raroi), conf)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to add borderline workflow")
        if needs_review:
            self.needs_review.add(wf)
        else:
            self.needs_review.discard(wf)
        return final_score, needs_review, conf

    # ------------------------------------------------------------------
    def process_borderline_candidates(
        self,
        evaluator: Callable[[str, Dict[str, Any]], float | Tuple[float, float]] | None = None,
    ) -> None:
        """Convenience wrapper around :meth:`BorderlineBucket.process`.

        Parameters
        ----------
        evaluator:
            Optional callable returning either ``raroi`` or ``(raroi, confidence)``
            for ``(workflow_id, candidate_info)``. When omitted the candidate's
            last recorded values are reused.
        """

        self.borderline_bucket.process(
            evaluator,
            raroi_threshold=self.raroi_borderline_threshold,
            confidence_threshold=self.confidence_threshold,
        )

    # ------------------------------------------------------------------
    def prediction_summary(self, window: int | None = None) -> Dict[str, Any]:
        """Return rolling error metrics and class stats for ``window``."""
        wf_mae = {
            wf: vals[-window:] if window else list(vals)
            for wf, vals in self.workflow_mae_history.items()
        }
        wf_var = {
            wf: vals[-window:] if window else list(vals)
            for wf, vals in self.workflow_variance_history.items()
        }
        wf_conf = {
            wf: vals[-window:] if window else list(vals)
            for wf, vals in self.workflow_confidence_history.items()
        }

        return {
            "mae": self.rolling_mae(window),
            "accuracy": self.classification_accuracy(window),
            "class_counts": self.class_counts(window),
            "confusion_matrix": self.confusion_matrix(window),
            "mae_trend": self.rolling_mae_trend(window),
            "accuracy_trend": self.rolling_accuracy_trend(window),
            "scenario_roi_deltas": dict(self.scenario_roi_deltas),
            "scenario_raroi": dict(self.scenario_raroi),
            "scenario_raroi_delta": dict(self.scenario_raroi_delta),
            "scenario_metrics_delta": dict(self.scenario_metrics_delta),
            "scenario_synergy_delta": dict(self.scenario_synergy_delta),
            "worst_scenario": self.biggest_drop()[0] if self.scenario_roi_deltas else None,
            "workflow_mae": wf_mae,
            "workflow_variance": wf_var,
            "workflow_confidence": wf_conf,
        }

    # ------------------------------------------------------------------
    def synergy_reliability(self, window: int | None = None) -> float:
        """Return the mean absolute error of ``synergy_roi`` forecasts.

        The method compares each stored prediction for ``synergy_roi`` with the
        corresponding actual value recorded via
        :meth:`record_metric_prediction`. The absolute differences are averaged
        over the given ``window`` (or the entire history when ``window`` is
        ``None``) giving ``sum(|p\_i - a\_i|) / n``. The result indicates how
        closely synergy ROI forecasts matched reality: lower is better.
        """

        return self.rolling_mae_metric("synergy_roi", window)

    # ------------------------------------------------------------------
    def rolling_mae_long_term_roi(
        self, window: int | None = None
    ) -> float:
        """Return MAE for long-term ROI predictions."""
        return self.rolling_mae_metric("long_term_roi", window)

    def long_term_roi_reliability(
        self, window: int | None = None, cv: int | None = None
    ) -> float:
        """Return reliability score for long-term ROI predictions."""
        return self.reliability(metric="long_term_roi", window=window, cv=cv)

    # ------------------------------------------------------------------
    def reliability(
        self,
        *,
        metric: str | None = None,
        window: int | None = None,
        cv: int | None = None,
    ) -> float:
        """Return reliability score for predictions.

        When ``cv`` is provided :meth:`cv_reliability` is returned. Otherwise the
        rolling mean absolute error is converted to a 0-1 reliability using
        ``1 / (1 + MAE)``.
        """

        if cv:
            score = self.cv_reliability(metric, cv)
            if np.isnan(score):
                return 0.0
            return max(0.0, min(1.0, score))
        if metric is None:
            preds = self.predicted_roi
            acts = self.actual_roi
        else:
            preds = self.predicted_metrics.get(metric, [])
            acts = self.actual_metrics.get(metric, [])
        if window:
            preds = preds[-window:]
            acts = acts[-window:]
        errors = [abs(p - a) for p, a in zip(preds, acts)]
        if not errors:
            err = 0.0
        else:
            err = self._ema(errors)
        return 1.0 / (1.0 + abs(err))

    # ------------------------------------------------------------------
    def cv_reliability(self, metric: str | None = None, cv: int = 3) -> float:
        """Return cross-validated reliability score for predictions.

        Parameters
        ----------
        metric:
            Optional metric name. When omitted the ROI predictions are
            evaluated.
        cv:
            Number of folds to use for K-fold cross-validation.
        """

        preds: List[float]
        acts: List[float]
        if metric is None:
            preds = self.predicted_roi
            acts = self.actual_roi
        else:
            preds = self.predicted_metrics.get(metric, [])
            acts = self.actual_metrics.get(metric, [])

        if len(preds) < 4 or len(acts) < 4:
            return 0.0

        X = np.array(preds, dtype=float).reshape(-1, 1)
        y = np.array(acts, dtype=float)

        try:
            from sklearn.model_selection import KFold, cross_val_score

            folds = min(cv, max(2, len(preds) // 2))
            if folds < 2:
                return 0.0
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
            return float(np.nanmean(scores))
        except Exception:
            try:
                model = LinearRegression().fit(X, y)
                return float(model.score(X, y))
            except Exception:
                return 0.0

    # ------------------------------------------------------------------
    def entropy_gain(self) -> float:
        """Return average ROI gain per unit entropy delta."""

        n = min(self.window, len(self.roi_history), len(self.entropy_delta_history))
        if n == 0:
            return 0.0
        roi_avg = sum(self.roi_history[-n:]) / n
        entropy_avg = sum(abs(x) for x in self.entropy_delta_history[-n:]) / n
        if entropy_avg == 0:
            return 0.0
        return roi_avg / entropy_avg

    def cache_correlations(self, correlations: Mapping[tuple[str, str], float]) -> None:
        """Record pairwise correlation values for later volatility analysis."""

        for pair, value in correlations.items():
            key = tuple(sorted(pair))
            self.correlation_history.setdefault(key, []).append(float(value))

    # ------------------------------------------------------------------
    def entropy_ceiling(self, threshold: float, window: int = 5) -> bool:
        """Return ``True`` when ROI per entropy delta falls below ``threshold``.

        The mean ratio of ROI delta to entropy delta is computed for the most
        recent ``window`` entries. Entries with zero entropy delta are skipped to
        avoid division by zero. If no valid ratios exist the ceiling is
        considered unmet.
        """

        n = min(window, len(self.roi_history), len(self.entropy_delta_history))
        if n == 0:
            return False
        ratios: List[float] = []
        for roi, ent in zip(self.roi_history[-n:], self.entropy_delta_history[-n:]):
            if ent == 0:
                continue
            ratios.append(abs(roi) / abs(ent))
        if not ratios:
            return False
        mean_ratio = sum(ratios) / len(ratios)
        return mean_ratio < threshold

    # ------------------------------------------------------------------
    def update(
        self,
        roi_before: float,
        roi_after: float,
        modules: Optional[List[str]] = None,
        resources: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, float]] = None,
        category: str | None = None,
        confidence: float | None = None,
        retrieval_metrics: Sequence[Dict[str, Any]] | None = None,
        profile_type: str | None = None,
    ) -> Tuple[Optional[int], List[float], bool, bool]:
        """Record ROI delta and evaluate stopping criteria.

        When ``modules`` is provided, track ROI contributions per module. When
        ``resources`` is given, CPU, memory, disk, time and GPU metrics are stored for
        use in forecasting. ``metrics`` allows arbitrary named metrics to be
        recorded for later forecasting with :meth:`forecast_metric`. ``category``
        captures the predicted ROI growth classification for this iteration and
        ``confidence`` records the model's confidence in the prediction.  When
        ``profile_type`` is supplied, low ROI scores or vetoes trigger bottleneck
        suggestions via :func:`propose_fix`.

        Returns a tuple of ``(vertex, predictions, should_stop, entropy_ceiling)``
        where ``entropy_ceiling`` is ``True`` when the ROI gain per entropy delta
        falls below the configured tolerance.
        """
        predicted_class = self._next_category
        if self._adaptive_predictor is None:
            try:
                from .adaptive_roi_predictor import AdaptiveROIPredictor

                self._adaptive_predictor = AdaptiveROIPredictor()
            except Exception:
                self._adaptive_predictor = None
        if metrics is None:
            metrics = {}
        metrics.setdefault("roi_reliability", self.reliability())
        metrics.setdefault(
            "synergy_roi_reliability", self.reliability(metric="synergy_roi")
        )
        metrics.setdefault("synergy_reliability", self.synergy_reliability())
        metrics.setdefault(
            "long_term_roi_reliability", self.long_term_roi_reliability()
        )

        delta = roi_after - roi_before
        filtered = self._filter_value(delta)
        if category is None and predicted_class is not None:
            category = predicted_class
        if category is not None:
            self.category_history.append(category)
        weight = 1.0
        if category == "exponential":
            weight = 1.5
        elif category == "marginal":
            weight = 0.5
        adjusted: float | None = None
        final_score = 0.0
        if filtered is not None:
            hist_for_pred = self.roi_history or [roi_before]
            if self._next_prediction is None:
                if self._adaptive_predictor is not None:
                    try:
                        feats = [[float(x)] for x in (hist_for_pred + [hist_for_pred[-1]])]
                        preds, cls, _, cls_conf = self._adaptive_predictor.predict(
                            feats, horizon=len(feats)
                        )
                        self._next_prediction = float(preds[-1][0])
                        predicted_class = predicted_class or cls
                        if confidence is None:
                            confidence = cls_conf
                    except Exception:
                        self._next_prediction = float(hist_for_pred[-1])
                else:
                    self._next_prediction = float(hist_for_pred[-1])
            adjusted = filtered * weight
            self.roi_history.append(adjusted)

            # Extract recent runtime metrics used for risk calculation
            errors_per_minute = float(metrics.get("errors_per_minute", 0.0))
            self._last_errors_per_minute = errors_per_minute

            # Collect critical test results if provided in metrics
            tests: Dict[str, bool] = {}
            if isinstance(metrics.get("test_status"), Mapping):
                tests = {
                    str(k): bool(v)
                    for k, v in metrics.get("test_status", {}).items()  # type: ignore[arg-type]
                }
            else:
                for key in CRITICAL_SUITES:
                    val = metrics.get(key)
                    if isinstance(val, bool):
                        tests[key] = val
            failing = [name for name, passed in tests.items() if not passed]
            self._last_test_failures = failing
            sts_module = _get_self_test_service()
            if sts_module is not None:
                try:
                    sts_module.set_failed_critical_tests(failing)
                except Exception:
                    pass

            rb: float | None
            try:
                rb = float(metrics["rollback_probability"])  # type: ignore[index]
            except Exception:
                rb = None

            _base_roi, raroi, _suggestions = self.calculate_raroi(
                adjusted,
                workflow_type=str(metrics.get("workflow_type", "standard")),
                rollback_prob=rb,
                metrics=metrics,
                failing_tests=failing,
            )
            self.raroi_history.append(raroi)
            self.confidence_history.append(float(confidence or 0.0))
            targets = ["_global"]
            if modules:
                targets.extend(str(m) for m in modules)
            if confidence is not None:
                for wf in targets:
                    if wf != "_global":
                        self.workflow_confidence_scores[str(wf)] = float(confidence)
            final_score, _nr, _conf = self.score_workflow(targets[0], raroi)
            for wf in targets[1:]:
                self.score_workflow(wf, raroi)
            if modules:
                for m in modules:
                    cid = self.cluster_map.get(m)
                    key = str(cid) if cid is not None else m
                    self.module_deltas.setdefault(key, []).append(adjusted)
                    self.module_raroi.setdefault(key, []).append(raroi)
                    if cid is not None:
                        self.cluster_deltas.setdefault(cid, []).append(adjusted)
                        self.cluster_raroi.setdefault(cid, []).append(raroi)
                if (
                    self.results_db is not None
                    and self.current_workflow_id is not None
                    and self.current_run_id is not None
                ):
                    runtime = 0.0
                    success_rate = 0.0
                    if metrics:
                        runtime = float(
                            metrics.get("workflow_runtime", metrics.get("runtime", 0.0))
                        )
                        success_rate = float(metrics.get("success_rate", 0.0))
                    for m in modules:
                        try:
                            self.results_db.log_module_delta(
                                self.current_workflow_id,
                                self.current_run_id,
                                str(m),
                                runtime,
                                success_rate,
                                float(adjusted),
                            )
                        except Exception:
                            pass
            if retrieval_metrics:
                hits = [m for m in retrieval_metrics if m.get("hit")]
                if hits:
                    tot = (
                        sum(float(m.get("tokens", 0.0)) for m in hits)
                        or float(len(hits))
                    )
                    for m in hits:
                        origin = str(m.get("origin_db", "unknown"))
                        tokens = float(m.get("tokens", 0.0))
                        w = tokens / tot if tot else 0.0
                        contrib = adjusted * (w or 1.0 / len(hits))
                        self._origin_db_deltas.setdefault(origin, []).append(contrib)
                        if _DB_ROI_GAUGE is not None:
                            avg = sum(self._origin_db_deltas[origin]) / len(
                                self._origin_db_deltas[origin]
                            )
                            _DB_ROI_GAUGE.labels(origin_db=origin).set(avg)
            if resources:
                try:
                    self.resource_metrics.append(
                        (
                            float(resources.get("cpu", 0.0)),
                            float(resources.get("memory", 0.0)),
                            float(resources.get("disk", 0.0)),
                            float(resources.get("time", 0.0)),
                            float(resources.get("gpu", 0.0)),
                        )
                    )
                except Exception:
                    self.resource_metrics.append((0.0, 0.0, 0.0, 0.0, 0.0))
            entropy_val: float | None = None
            if metrics:
                if "synergy_shannon_entropy" in metrics:
                    try:
                        entropy_val = float(metrics["synergy_shannon_entropy"])
                    except Exception:
                        entropy_val = 0.0
                elif "shannon_entropy" in metrics:
                    try:
                        entropy_val = float(metrics["shannon_entropy"])
                    except Exception:
                        entropy_val = 0.0
            if entropy_val is None:
                entropy_val = self.entropy_history[-1] if self.entropy_history else 0.0
            prev_entropy = self.entropy_history[-1] if self.entropy_history else entropy_val
            entropy_delta = entropy_val - prev_entropy
            self.entropy_history.append(entropy_val)
            self.entropy_delta_history.append(entropy_delta)
            if modules:
                for m in modules:
                    cid = self.cluster_map.get(m)
                    key = str(cid) if cid is not None else m
                    self.module_entropy_deltas.setdefault(key, []).append(entropy_delta)
            if metrics:
                for name, value in metrics.items():
                    try:
                        val = float(value)
                    except Exception:
                        val = 0.0
                    target = (
                        self.synergy_metrics_history
                        if str(name).startswith("synergy_")
                        else self.metrics_history
                    )
                    target.setdefault(str(name), []).append(val)
                    if str(name).startswith("synergy_"):
                        self.metrics_history.setdefault(str(name), []).append(val)
            for name in list(self.metrics_history):
                if not metrics or name not in metrics:
                    last = (
                        self.metrics_history[name][-1]
                        if self.metrics_history[name]
                        else 0.0
                    )
                    self.metrics_history[name].append(last)
            for name in list(self.synergy_metrics_history):
                if not metrics or name not in metrics:
                    last = (
                        self.synergy_metrics_history[name][-1]
                        if self.synergy_metrics_history[name]
                        else 0.0
                    )
                    self.synergy_metrics_history[name].append(last)

        if self._next_prediction is not None:
            self.record_prediction(
                self._next_prediction,
                roi_after,
                predicted_class=predicted_class,
                confidence=confidence,
                modules=modules,
                final_score=final_score,
            )
        self._next_prediction = None
        self._next_category = None

        mae_val = self.rolling_mae()
        roi_var_val = float(np.var(self.actual_roi)) if self.actual_roi else 0.0
        if filtered is not None:
            self.mae_history.append(mae_val)
            self.variance_history.append(roi_var_val)
        if _me is not None:
            try:
                _me.roi_confidence.set(float(confidence or 0.0))
                _me.roi_mae.set(mae_val)
                _me.roi_variance.set(roi_var_val)
            except Exception:
                pass
        logger.info(
            "roi update",
            extra=log_record(
                delta=delta,
                category=category,
                adjusted=adjusted,
                confidence=confidence,
                mae=mae_val,
                roi_variance=roi_var_val,
                final_score=final_score,
            ),
        )

        if profile_type is not None and metrics:
            try:
                calc = ROICalculator()
                result = calc.calculate(metrics, profile_type)
                score, vetoed = result.score, result.vetoed
                if score < self.raroi_borderline_threshold or vetoed:
                    suggestions = propose_fix(metrics, profile_type)
                    if suggestions:
                        logger.info(
                            "roi bottlenecks",
                            extra=log_record(suggestions=suggestions),
                        )
                        if _ROI_BOTTLENECK_GAUGE is not None:
                            for metric, hint in suggestions:
                                try:
                                    _ROI_BOTTLENECK_GAUGE.labels(
                                        metric=metric, hint=hint
                                    ).set(1)
                                except Exception:
                                    pass
            except Exception:
                logger.exception("bottleneck suggestion failed")

        iteration = len(self.roi_history) - 1
        vertex, preds = self._regression()
        should_stop = False
        if vertex is not None and iteration >= vertex:
            should_stop = True
        if len(self.roi_history) >= self.window:
            if self.weights is None:
                avg = sum(self.roi_history[-self.window :]) / self.window
            else:
                data = np.array(self.roi_history[-self.window :], dtype=float)
                avg = float(np.dot(data, self.weights))
            if abs(avg) < self.tolerance:
                should_stop = True
        # predicted gain-based termination when recent categories are marginal
        if (
            len(self.category_history) >= self.window
            and all(c == "marginal" for c in self.category_history[-self.window:])
        ):
            predicted_gain, _ = self.forecast()
            if abs(predicted_gain) < self.tolerance:
                should_stop = True
        if len(self.roi_history) % self.evaluate_every == 0:
            try:
                if self._adaptive_predictor is not None:
                    self._adaptive_predictor.evaluate_model(
                        self,
                        window=self.evaluation_window,
                        mae_threshold=self.mae_threshold,
                        accuracy_threshold=self.acc_threshold,
                    )
                else:
                    self.evaluate_model(
                        window=self.evaluation_window,
                        mae_threshold=self.mae_threshold,
                        acc_threshold=self.acc_threshold,
                    )
            except Exception:
                logger.exception("model evaluation failed")
        entropy_stop = self.entropy_ceiling(
            self.entropy_threshold, window=self.window
        )
        return vertex, preds, should_stop, entropy_stop

    # ------------------------------------------------------------------
    def final_score(self, workflow: str) -> Tuple[float, bool]:
        """Return latest final score and review status for ``workflow``."""

        hist = self.final_roi_history.get(str(workflow)) or []
        score = hist[-1] if hist else 0.0
        return score, str(workflow) in self.needs_review

    # ------------------------------------------------------------------
    def forecast(self) -> Tuple[float, Tuple[float, float]]:
        """Return next ROI prediction and 95% confidence interval.

        When ``statsmodels`` is available multiple ARIMA orders are evaluated
        using ``aic`` and ``bic`` scores. The best order is cached until the
        history changes. If ``statsmodels`` is missing or fitting fails, a
        simple linear regression forecast is used instead.
        """
        if self._adaptive_predictor is not None:
            try:
                feats = [[float(x)] for x in self.roi_history] or [[0.0]]
                try:
                    _, cls, _, _ = self._adaptive_predictor.predict(feats, horizon=len(feats))
                except TypeError:
                    _, cls, _, _ = self._adaptive_predictor.predict(feats)
                self._next_category = cls
            except Exception:
                self._next_category = None
        else:
            self._next_category = None

        if not self.roi_history:
            self._next_prediction = 0.0
            return 0.0, (0.0, 0.0)
        if len(self.roi_history) < 2:
            val = float(self.roi_history[-1])
            self._next_prediction = val
            return val, (val, val)
        history_tuple = tuple(self.roi_history)
        exog = None
        if self.resource_metrics:
            arr = np.array(self.resource_metrics, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= len(self.roi_history):
                exog = arr[-len(self.roi_history) :]

        if os.getenv("ENABLE_ADVANCED_ROI_PREDICTOR") == "1":
            try:
                from .roi_predictor import ROIPredictor

                predictor = ROIPredictor()
                mean, (lower, upper) = predictor.forecast(self.roi_history, exog=exog)
                self._next_prediction = mean
                return mean, (lower, upper)
            except Exception:
                logger.exception("advanced ROI predictor failed")
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore

            if self._best_order is None or self._order_history != history_tuple:
                candidates = [
                    (1, 1, 1),
                    (1, 0, 0),
                    (0, 1, 1),
                    (2, 1, 2),
                ]
                scores = []
                for order in candidates:
                    try:
                        m = ARIMA(
                            self.roi_history,
                            exog=exog,
                            order=order,
                        ).fit()
                        scores.append((m.aic, m.bic, order, m))
                    except Exception:
                        logger.exception("candidate ARIMA order failed")
                        continue
                if scores:
                    scores.sort(key=lambda x: (x[0], x[1]))
                    _, _, self._best_order, model = scores[0]
                    self._order_history = history_tuple
                else:
                    model = ARIMA(self.roi_history, order=(1, 1, 1)).fit()
                    self._best_order = (1, 1, 1)
                    self._order_history = history_tuple
            else:
                model = ARIMA(
                    self.roi_history,
                    exog=exog,
                    order=self._best_order,
                ).fit()

            next_exog = exog[-1:] if exog is not None else None
            res = model.get_forecast(steps=1, exog=next_exog)
            mean = float(res.predicted_mean)
            conf = res.conf_int(alpha=0.05)[0]
            lower, upper = float(conf[0]), float(conf[1])
            self._next_prediction = mean
            return mean, (lower, upper)
        except Exception:
            logger.exception("ARIMA forecast failed")
        try:
            X = np.arange(len(self.roi_history)).reshape(-1, 1)
            y = np.array(self.roi_history)
            if exog is not None and exog.shape[0] >= len(self.roi_history):
                X = np.hstack([X, exog[-len(self.roi_history) :]])
                next_row = np.hstack([[len(y)], exog[-1]]).reshape(1, -1)
            else:
                next_row = [[len(y)]]
            lr = LinearRegression().fit(X, y)
            mean = float(lr.predict(next_row)[0])
            # Standard error of residuals for naive confidence interval
            resid = y - lr.predict(X)
            if resid.size > 1:
                se = float(resid.std(ddof=1))
            else:
                se = 0.0
            delta = 1.96 * se
            self._next_prediction = mean
            return mean, (mean - delta, mean + delta)
        except Exception:
            logger.exception("linear regression forecast failed")
            val = float(self.roi_history[-1])
            self._next_prediction = val
            return val, (val, val)

    # ------------------------------------------------------------------
    def _forecast_generic(
        self, history: List[float]
    ) -> Tuple[float, Tuple[float, float]]:
        """Internal helper to forecast the next value for ``history``."""
        if not history:
            return 0.0, (0.0, 0.0)
        if len(history) < 2:
            val = float(history[-1])
            return val, (val, val)
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore

            model = ARIMA(history, order=(1, 1, 1)).fit()
            res = model.get_forecast(steps=1)
            mean = float(res.predicted_mean)
            conf = res.conf_int(alpha=0.05)[0]
            lower, upper = float(conf[0]), float(conf[1])
            return mean, (lower, upper)
        except Exception:
            logger.exception("ARIMA forecast failed")
        try:
            X = np.arange(len(history)).reshape(-1, 1)
            y = np.array(history)
            lr = LinearRegression().fit(X, y)
            mean = float(lr.predict(np.array([[len(history)]]))[0])
            resid = y - lr.predict(X)
            if resid.size > 1:
                se = float(resid.std(ddof=1))
            else:
                se = 0.0
            delta = 1.96 * se
            return mean, (mean - delta, mean + delta)
        except Exception:
            logger.exception("linear regression forecast failed")
            val = float(history[-1])
            return val, (val, val)

    # ------------------------------------------------------------------
    def forecast_metric(self, metric_name: str) -> Tuple[float, Tuple[float, float]]:
        """Return forecast for a recorded metric series.

        ``metric_name`` may start with ``"synergy_"`` to access synergy metrics
        generated by the sandbox. In this case the underlying series is looked
        up by name without modification, falling back to the base metric when no
        synergy history exists.
        """

        name = str(metric_name)
        history = self.metrics_history.get(name)
        if history is None and name.startswith("synergy_"):
            history = self.synergy_metrics_history.get(name)
            if history is None:
                history = self.metrics_history.get(name[len("synergy_") :])
        if history is None:
            history = []
        return self._forecast_generic(history)

    # ------------------------------------------------------------------
    def forecast_synergy(self) -> Tuple[float, Tuple[float, float]]:
        """Return forecast for ``synergy_roi`` using recorded history."""

        history = self.synergy_metrics_history.get("synergy_roi")
        if history is None:
            history = self.metrics_history.get("synergy_roi")
        if history is None:
            history = []

        if len(history) > 1:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

                model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False)
                res = model.get_forecast(steps=1)
                mean = float(res.predicted_mean[0])
                conf = res.conf_int(alpha=0.05).iloc[0]
                lower, upper = float(conf[0]), float(conf[1])
                return mean, (lower, upper)
            except Exception:
                logger.exception("SARIMAX synergy forecast failed")

        return self._forecast_generic(history)

    def forecast_synergy_metric(self, name: str) -> Tuple[float, Tuple[float, float]]:
        """Return forecast for ``synergy_<name>`` history."""

        metric_name = (
            f"synergy_{name}" if not str(name).startswith("synergy_") else str(name)
        )
        history = self.synergy_metrics_history.get(metric_name)
        if history is None:
            history = self.metrics_history.get(metric_name)
        if method != "arima" and history:
            n = min(len(history), window)
            return float(round(self._ema(history[-n:]), 4))
        if history is None:
            history = []
        return self._forecast_generic(history)

    # ------------------------------------------------------------------
    def predict_metric_with_manager(
        self,
        manager: "PredictionManager",
        metric: str,
        features: Iterable[float] | None = None,
        *,
        actual: float | None = None,
        bot_name: str | None = None,
    ) -> float:
        """Return averaged ``metric`` prediction from ``manager``.

        When ``actual`` is provided the prediction is also recorded via
        :meth:`record_metric_prediction`.
        """

        bots = manager.get_prediction_bots_for(bot_name or "roi_tracker")
        if not bots:
            try:
                bots = manager.assign_prediction_bots(self)
            except Exception:
                bots = []

        vec = list(features or [])
        score = 0.0
        count = 0
        for bid in bots:
            entry = manager.registry.get(bid)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict_metric", None)
            if not callable(pred):
                continue
            try:
                val = pred(metric, vec)
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception:
                continue

        pred_value = float(score / count) if count else 0.0
        if actual is not None:
            try:
                self.record_metric_prediction(metric, pred_value, float(actual))
            except Exception:
                logger.exception("prediction manager failed")
        return pred_value

    # ------------------------------------------------------------------
    def predict_all_metrics(
        self,
        manager: "PredictionManager",
        features: Iterable[float] | None = None,
        *,
        bot_name: str | None = None,
    ) -> Dict[str, float]:
        """Predict all recorded metrics via ``manager`` and record results."""
        if manager:
            try:
                manager.assign_prediction_bots(self)
            except Exception:
                logger.exception("manager assignment failed")

        results: Dict[str, float] = {}
        vec = list(features or [])
        for name, history in list(self.metrics_history.items()):
            if name.startswith("synergy_"):
                continue
            if name in ("roi_reliability", "long_term_roi_reliability"):
                continue
            if not history and name not in self.predicted_metrics:
                continue
            if history and all(float(v) == 0.0 for v in history) and name not in self.predicted_metrics:
                continue
            actual = history[-1] if history else None
            try:
                pred = self.predict_metric_with_manager(
                    manager,
                    name,
                    vec,
                    actual=actual,
                    bot_name=bot_name,
                )
            except Exception:
                pred = 0.0
            results[name] = float(pred)
        return results

    # ------------------------------------------------------------------
    def predict_synergy(self, window: int = 5) -> float:
        """Return forecast for ``synergy_roi`` using ROI and metric histories."""

        history = self.metrics_history.get("synergy_roi")
        if not history or len(history) < 2:
            return 0.0
        method = os.getenv("SYNERGY_FORECAST_METHOD", "ema").lower()
        if method != "arima":
            n = min(len(history), window)
            return float(round(self._ema(history[-n:]), 4))

        model_name = os.getenv("SANDBOX_SYNERGY_MODEL")
        if model_name and len(history) > 10:
            try:
                from . import synergy_predictor as sp

                if model_name.lower() == "arima":
                    return float(sp.ARIMASynergyPredictor().predict(history))
                if model_name.lower() == "lstm":
                    if getattr(sp, "torch", None) is None:
                        return float(sp.ARIMASynergyPredictor().predict(history))
                    return float(sp.LSTMSynergyPredictor().predict(history))
            except Exception:
                logger.exception("synergy predictor failed")

        n = min(len(history), window, len(self.roi_history))
        metrics = [m for m in self.metrics_history if not m.startswith("synergy_")]

        X: list[list[float]] = []
        y: list[float] = []
        for i in range(-n, 0):
            row = [float(self.roi_history[i])]
            for m in metrics:
                vals = self.metrics_history.get(m, [])
                if len(vals) >= len(self.roi_history):
                    row.append(float(vals[i]))
                elif vals:
                    row.append(float(vals[-1]))
                else:
                    row.append(0.0)
            y.append(float(history[i]))
            X.append(row)

        if len(X) < 2:
            return float(history[-1])

        arr_X = np.array(X, dtype=float)
        arr_y = np.array(y, dtype=float)

        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore

            model = ARIMA(arr_y, exog=arr_X, order=(1, 1, 1)).fit()
            roi_pred, _ = self.forecast()
            next_row = [roi_pred]
            for m in metrics:
                pred, _ = self.forecast_metric(m)
                next_row.append(pred)
            res = model.get_forecast(steps=1, exog=np.array([next_row]))
            return float(round(float(res.predicted_mean[0]), 4))
        except Exception:
            logger.exception("ARIMA synergy forecast failed")

        try:
            model = LinearRegression().fit(arr_X, arr_y)
            roi_pred, _ = self.forecast()
            next_row = [roi_pred]
            for m in metrics:
                pred, _ = self.forecast_metric(m)
                next_row.append(pred)
            return float(round(float(model.predict([next_row])[0]), 4))
        except Exception:
            logger.exception("linear regression synergy forecast failed")
            return float(history[-1])

    # ------------------------------------------------------------------
    def predict_synergy_metric(
        self,
        name: str,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_<name>`` using ROI and metric histories.

        When ``manager`` is provided the prediction bots registered with
        :class:`PredictionManager` are consulted before falling back to the
        internal statistical model.
        """

        metric_name = (
            f"synergy_{name}" if not str(name).startswith("synergy_") else str(name)
        )
        history = self.synergy_metrics_history.get(metric_name)
        if history is None:
            history = self.metrics_history.get(metric_name)
        method = os.getenv("SYNERGY_FORECAST_METHOD", "ema").lower()
        if method != "arima" and history:
            n = min(len(history), window)
            return float(round(self._ema(history[-n:]), 4))

        if manager is not None:
            try:
                actual = history[-1] if history else None
                val = self.predict_metric_with_manager(
                    manager, metric_name, [], actual=actual
                )
                return float(val)
            except Exception:
                logger.exception("synergy prediction via manager failed")

        model_name = os.getenv("SANDBOX_SYNERGY_MODEL")
        if model_name and len(history) > 10:
            try:
                from . import synergy_predictor as sp

                if model_name.lower() == "arima":
                    return float(sp.ARIMASynergyPredictor().predict(history))
                if model_name.lower() == "lstm":
                    if getattr(sp, "torch", None) is None:
                        return float(sp.ARIMASynergyPredictor().predict(history))
                    return float(sp.LSTMSynergyPredictor().predict(history))
            except Exception:
                logger.exception("synergy predictor failed")

        if not history or len(history) < 2:
            return 0.0

        n = min(len(history), window, len(self.roi_history))
        metrics = [m for m in self.metrics_history if not m.startswith("synergy_")]

        X: list[list[float]] = []
        y: list[float] = []
        for i in range(-n, 0):
            row = [float(self.roi_history[i])]
            for m in metrics:
                vals = self.metrics_history.get(m, [])
                if len(vals) >= len(self.roi_history):
                    row.append(float(vals[i]))
                elif vals:
                    row.append(float(vals[-1]))
                else:
                    row.append(0.0)
            y.append(float(history[i]))
            X.append(row)

        if len(X) < 2:
            return float(history[-1])

        arr_X = np.array(X, dtype=float)
        arr_y = np.array(y, dtype=float)

        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore

            model = ARIMA(arr_y, exog=arr_X, order=(1, 1, 1)).fit()
            roi_pred, _ = self.forecast()
            next_row = [roi_pred]
            for m in metrics:
                pred, _ = self.forecast_metric(m)
                next_row.append(pred)
            res = model.get_forecast(steps=1, exog=np.array([next_row]))
            return float(round(float(res.predicted_mean[0]), 4))
        except Exception:
            logger.exception("ARIMA synergy forecast failed")

        try:
            model = LinearRegression().fit(arr_X, arr_y)
            roi_pred, _ = self.forecast()
            next_row = [roi_pred]
            for m in metrics:
                pred, _ = self.forecast_metric(m)
                next_row.append(pred)
            return float(round(float(model.predict([next_row])[0]), 4))
        except Exception:
            logger.exception("linear regression synergy forecast failed")
            return float(history[-1])

    # ------------------------------------------------------------------
    def predict_synergy_profitability(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_profitability``."""
        return self.predict_synergy_metric("profitability", window, manager)

    def predict_synergy_revenue(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_revenue``."""
        return self.predict_synergy_metric("revenue", window, manager)

    def predict_synergy_projected_lucrativity(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_projected_lucrativity``."""
        return self.predict_synergy_metric("projected_lucrativity", window, manager)

    def predict_synergy_maintainability(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_maintainability``."""
        return self.predict_synergy_metric("maintainability", window, manager)

    def predict_synergy_adaptability(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_adaptability``."""
        return self.predict_synergy_metric("adaptability", window, manager)

    def predict_synergy_code_quality(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_code_quality``."""
        return self.predict_synergy_metric("code_quality", window, manager)

    def predict_synergy_safety_rating(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_safety_rating``."""
        return self.predict_synergy_metric("safety_rating", window, manager)

    def predict_synergy_network_latency(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_network_latency``."""
        return self.predict_synergy_metric("network_latency", window, manager)

    def predict_synergy_throughput(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_throughput``."""
        return self.predict_synergy_metric("throughput", window, manager)

    def predict_synergy_risk_index(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_risk_index``."""
        return self.predict_synergy_metric("risk_index", window, manager)

    def predict_synergy_recovery_time(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_recovery_time``."""
        return self.predict_synergy_metric("recovery_time", window, manager)

    def predict_synergy_discrepancy_count(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_discrepancy_count``."""
        return self.predict_synergy_metric("discrepancy_count", window, manager)

    def predict_synergy_gpu_usage(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_gpu_usage``."""
        return self.predict_synergy_metric("gpu_usage", window, manager)

    def predict_synergy_cpu_usage(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_cpu_usage``."""
        return self.predict_synergy_metric("cpu_usage", window, manager)

    def predict_synergy_memory_usage(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_memory_usage``."""
        return self.predict_synergy_metric("memory_usage", window, manager)

    def predict_synergy_long_term_lucrativity(
        self,
        window: int = 5,
        manager: "PredictionManager | None" = None,
    ) -> float:
        """Return forecast for ``synergy_long_term_lucrativity``."""
        return self.predict_synergy_metric("long_term_lucrativity", window, manager)

    # ------------------------------------------------------------------
    def get_scenario_synergy(self, name: str) -> List[Dict[str, float]]:
        """Return recorded synergy metrics for ``name`` scenario."""

        return list(self.scenario_synergy.get(str(name), []))

    # ------------------------------------------------------------------
    def record_scenario_delta(
        self,
        name: str,
        delta: float,
        metrics_delta: Mapping[str, float] | None = None,
        synergy_delta: Mapping[str, float] | None = None,
        raroi: float | None = None,
        raroi_delta: float | None = None,
    ) -> None:
        """Record ROI ``delta`` and related information for scenario ``name``.

        Parameters
        ----------
        name:
            Scenario identifier.
        delta:
            ROI delta for the scenario.
        metrics_delta:
            Optional mapping of metric deltas relative to the baseline run.
        synergy_delta:
            Optional mapping capturing the difference between workflow-on and
            workflow-off runs for each synergy metric.
        raroi:
            Risk-adjusted ROI for the scenario.
        raroi_delta:
            RAROI delta relative to the baseline scenario.
        """

        scen = str(name)
        delta = float(delta)
        self.scenario_roi_deltas[scen] = delta
        if raroi is not None:
            self.scenario_raroi[scen] = float(raroi)
        if raroi_delta is not None:
            self.scenario_raroi_delta[scen] = float(raroi_delta)
        if metrics_delta is not None:
            self.scenario_metrics_delta[scen] = {
                str(k): float(v) for k, v in metrics_delta.items()
            }
        if synergy_delta is not None:
            self.scenario_synergy_delta[scen] = {
                str(k): float(v) for k, v in synergy_delta.items()
            }
        if (
            self._worst_scenario is None
            or delta < self.scenario_roi_deltas.get(self._worst_scenario, float("inf"))
        ):
            self._worst_scenario = scen

    # ------------------------------------------------------------------
    def biggest_drop(self) -> Tuple[str, float]:
        """Return the scenario with the largest negative ROI delta."""

        if not self.scenario_roi_deltas:
            return "", 0.0
        worst = min(self.scenario_roi_deltas.items(), key=lambda kv: kv[1])
        self._worst_scenario = worst[0]
        return worst

    # ------------------------------------------------------------------
    def worst_scenario(self) -> Tuple[str, float]:
        """Alias for :meth:`biggest_drop` for backward compatibility."""

        return self.biggest_drop()

    # ------------------------------------------------------------------
    def scenario_degradation(self) -> float:
        """Return ROI delta of the worst recorded scenario."""

        _, delta = self.biggest_drop()
        return float(delta)

    # ------------------------------------------------------------------
    def get_scenario_roi_delta(self, name: str) -> float:
        """Return ROI delta recorded for scenario ``name``."""

        return float(self.scenario_roi_deltas.get(str(name), 0.0))

    # ------------------------------------------------------------------
    def get_scenario_metrics_delta(self, name: str) -> Dict[str, float]:
        """Return metrics delta mapping recorded for ``name`` scenario."""

        return dict(self.scenario_metrics_delta.get(str(name), {}))

    # ------------------------------------------------------------------
    def get_scenario_synergy_delta(self, name: str) -> Dict[str, float]:
        """Return synergy difference mapping recorded for ``name`` scenario."""

        return dict(self.scenario_synergy_delta.get(str(name), {}))

    # ------------------------------------------------------------------
    def get_scenario_raroi_delta(self, name: str) -> float:
        """Return RAROI delta recorded for scenario ``name``."""

        return float(self.scenario_raroi_delta.get(str(name), 0.0))

    # ------------------------------------------------------------------
    def generate_scorecards(self) -> List[ScenarioScorecard]:
        """Return a list of :class:`ScenarioScorecard` summarising scenarios."""

        cards: List[ScenarioScorecard] = []
        threshold = 0.0
        failing: List[str] = []
        passing: List[str] = []
        for scen, delta in self.scenario_roi_deltas.items():
            metrics = self.scenario_metrics_delta.get(scen, {})
            triggers_failure = any(
                (("failure" in k) or ("error" in k) or k.endswith("_breach"))
                and v > 0
                for k, v in metrics.items()
            )
            if delta < threshold or triggers_failure:
                failing.append(scen)
            else:
                passing.append(scen)
        if len(failing) == 1 and passing:
            self.workflow_label = "situationally weak"
        else:
            self.workflow_label = None
        for scen, roi_delta in self.scenario_roi_deltas.items():
            cards.append(
                ScenarioScorecard(
                    scenario=scen,
                    roi_delta=float(roi_delta),
                    raroi_delta=float(self.scenario_raroi_delta.get(scen, 0.0)),
                    metrics_delta=dict(self.scenario_metrics_delta.get(scen, {})),
                    synergy_delta=dict(self.scenario_synergy_delta.get(scen, {})),
                    recommendation=HARDENING_TIPS.get(scen),
                    status=self.workflow_label,
                    score=float(self.scenario_raroi_delta.get(scen, 0.0)),
                )
            )
        return cards

    # ------------------------------------------------------------------
    def generate_scenario_scorecard(
        self,
        workflow_id: str,
        scenarios: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Return a mapping summarising stress test deltas per scenario.

        Parameters
        ----------
        workflow_id:
            Identifier of the workflow the scorecard belongs to.
        scenarios:
            Optional explicit list of scenario names. When omitted the
            presets from :func:`sandbox_runner.environment.default_scenario_presets`
            are used if available, otherwise all tracked scenarios are
            included.

        Returns
        -------
        dict
            Mapping containing the ``workflow_id`` and per-scenario deltas
            for ROI, metrics and synergy.
        """

        names: Sequence[str]
        if scenarios is None:
            try:  # pragma: no cover - optional dependency
                from .sandbox_runner import environment as _env  # type: ignore

                names = [
                    p.get("SCENARIO_NAME")
                    for p in _env.default_scenario_presets()
                    if p.get("SCENARIO_NAME")
                ]
            except Exception:  # pragma: no cover - best effort
                names = list(self.scenario_roi_deltas.keys())
        else:
            names = list(scenarios)

        card: Dict[str, Any] = {"workflow_id": str(workflow_id), "scenarios": {}}
        scorecards = {c.scenario: c for c in self.generate_scorecards()}
        if self.workflow_label:
            card["status"] = self.workflow_label
        for scen in names:
            score_val = float(self.scenario_raroi_delta.get(scen, 0.0))
            entry = {
                "roi_delta": float(self.scenario_roi_deltas.get(scen, 0.0)),
                "raroi_delta": score_val,
                "metrics_delta": dict(self.scenario_metrics_delta.get(scen, {})),
                "synergy_delta": dict(self.scenario_synergy_delta.get(scen, {})),
                "score": score_val,
            }
            rec = scorecards.get(scen)
            if rec:
                if rec.recommendation:
                    entry["recommendation"] = rec.recommendation
                if rec.status:
                    entry["status"] = rec.status
            card["scenarios"][scen] = entry
        return card

    # ------------------------------------------------------------------
    def build_governance_scorecard(
        self, workflow_id: str, scenarios: Sequence[str] | None = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Return ``(scorecard, metrics)`` for deployment governance evaluation.

        The returned ``scorecard`` mirrors :meth:`generate_scenario_scorecard`
        but also includes the latest RAROI, confidence and predicted ROI values.
        ``metrics`` exposes these values alongside a ``scenario_scores`` mapping
        suitable for :func:`deployment_governance.evaluate`.
        """

        card = self.generate_scenario_scorecard(workflow_id, scenarios)
        scenario_scores = {
            name: info.get("score", 0.0)
            for name, info in card.get("scenarios", {}).items()
        }
        card["raroi"] = self.last_raroi
        card["confidence"] = self.last_confidence
        card["predicted_roi"] = self.last_predicted_roi
        card["scenario_scores"] = scenario_scores
        metrics = {
            "raroi": self.last_raroi,
            "confidence": self.last_confidence,
            "predicted_roi": self.last_predicted_roi,
            "scenario_scores": scenario_scores,
        }
        return card, metrics

    def save_history(self, path: str) -> None:
        """Persist ``roi_history`` and ``module_deltas`` to ``path``."""
        if path.endswith(".json"):
            metric_preds = {
                m: [
                    [float(p), float(a)]
                    for p, a in zip(
                        self.predicted_metrics.get(m, []),
                        self.actual_metrics.get(m, []),
                    )
                ]
                for m in set(self.predicted_metrics) | set(self.actual_metrics)
            }
            data = {
                "roi_history": self.roi_history,
                "raroi_history": self.raroi_history,
                "confidence_history": self.confidence_history,
                "module_deltas": self.module_deltas,
                "module_raroi": self.module_raroi,
                "module_entropy_deltas": self.module_entropy_deltas,
                "origin_db_deltas": self._origin_db_deltas,
                "db_roi_metrics": self.db_roi_metrics,
                "predicted_roi": self.predicted_roi,
                "actual_roi": self.actual_roi,
                "category_history": self.category_history,
                "drift_scores": self.drift_scores,
                "drift_flags": self.drift_flags,
                "metrics_history": self.metrics_history,
                "synergy_metrics_history": self.synergy_metrics_history,
                "synergy_history": self.synergy_history,
                "scenario_synergy": self.scenario_synergy,
                "scenario_roi_deltas": self.scenario_roi_deltas,
                "scenario_raroi": self.scenario_raroi,
                "scenario_raroi_delta": self.scenario_raroi_delta,
                "scenario_metrics_delta": self.scenario_metrics_delta,
                "scenario_synergy_delta": self.scenario_synergy_delta,
                "worst_scenario": self.biggest_drop()[0] if self.scenario_roi_deltas else None,
                "predicted_metrics": self.predicted_metrics,
                "actual_metrics": self.actual_metrics,
                "metric_predictions": metric_preds,
                "drift_metrics": self.drift_metrics,
                "truth_adapter": self.truth_adapter.metadata if self.truth_adapter else {},
                "workflow_mae_history": {
                    k: list(v) for k, v in self.workflow_mae_history.items()
                },
                "workflow_variance_history": {
                    k: list(v) for k, v in self.workflow_variance_history.items()
                },
                "workflow_confidence_history": {
                    k: list(v) for k, v in self.workflow_confidence_history.items()
                },
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            return
        with router.get_connection("roi_history") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS roi_history (delta REAL, confidence REAL, raroi REAL, mae REAL, roi_variance REAL, final_score REAL)"
            )
            conn.execute("DELETE FROM roi_history")
            global_final = self.final_roi_history.get("_global", [])
            conn.executemany(
                "INSERT INTO roi_history (delta, confidence, raroi, mae, roi_variance, final_score) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        float(d),
                        float(
                            self.confidence_history[i]
                            if i < len(self.confidence_history)
                            else 0.0
                        ),
                        float(
                            self.raroi_history[i]
                            if i < len(self.raroi_history)
                            else 0.0
                        ),
                        float(
                            self.mae_history[i]
                            if i < len(self.mae_history)
                            else 0.0
                        ),
                        float(
                            self.variance_history[i]
                            if i < len(self.variance_history)
                            else 0.0
                        ),
                        float(
                            global_final[i]
                            if i < len(global_final)
                            else 0.0
                        ),
                    )
                    for i, d in enumerate(self.roi_history)
                ],
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS module_deltas (module TEXT, delta REAL, raroi REAL)"
            )
            conn.execute("DELETE FROM module_deltas")
            rows = []
            for m, vals in self.module_deltas.items():
                r_vals = self.module_raroi.get(m, [])
                for i, v in enumerate(vals):
                    rv = r_vals[i] if i < len(r_vals) else 0.0
                    rows.append((m, float(v), float(rv)))
            if rows:
                conn.executemany(
                    "INSERT INTO module_deltas (module, delta, raroi) VALUES (?, ?, ?)",
                    rows,
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS module_entropy_deltas (module TEXT, delta REAL)"
            )
            conn.execute("DELETE FROM module_entropy_deltas")
            ent_rows = [
                (m, float(v))
                for m, vals in self.module_entropy_deltas.items()
                for v in vals
            ]
            if ent_rows:
                conn.executemany(
                    "INSERT INTO module_entropy_deltas (module, delta) VALUES (?, ?)",
                    ent_rows,
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS db_deltas (origin_db TEXT, delta REAL)"
            )
            conn.execute("DELETE FROM db_deltas")
            db_rows = [
                (d, float(v))
                for d, vals in self._origin_db_deltas.items()
                for v in vals
            ]
            if db_rows:
                conn.executemany(
                    "INSERT INTO db_deltas (origin_db, delta) VALUES (?, ?)",
                    db_rows,
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS db_roi_metrics (
                    origin_db TEXT,
                    win_rate REAL,
                    regret_rate REAL,
                    roi REAL
                )
                """
            )
            conn.execute("DELETE FROM db_roi_metrics")
            if self.db_roi_metrics:
                conn.executemany(
                    "INSERT INTO db_roi_metrics (origin_db, win_rate, regret_rate, roi) VALUES (?,?,?,?)",
                    [
                        (
                            db,
                            float(m.get("win_rate", 0.0)),
                            float(m.get("regret_rate", 0.0)),
                            float(m.get("roi", 0.0)),
                        )
                        for db, m in self.db_roi_metrics.items()
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS predictions (pred REAL, actual REAL)"
            )
            conn.execute("DELETE FROM predictions")
            if self.predicted_roi:
                conn.executemany(
                    "INSERT INTO predictions (pred, actual) VALUES (?, ?)",
                    [
                        (float(p), float(a))
                        for p, a in zip(self.predicted_roi, self.actual_roi)
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS categories (category TEXT)"
            )
            conn.execute("DELETE FROM categories")
            if self.category_history:
                conn.executemany(
                    "INSERT INTO categories (category) VALUES (?)",
                    [(str(c),) for c in self.category_history],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS metric_predictions (metric TEXT, pred REAL, actual REAL)"
            )
            conn.execute("DELETE FROM metric_predictions")
            metric_pred_rows = [
                (m, float(p), float(a))
                for m, preds in self.predicted_metrics.items()
                for p, a in zip(preds, self.actual_metrics.get(m, []))
            ]
            if metric_pred_rows:
                conn.executemany(
                    "INSERT INTO metric_predictions (metric, pred, actual) VALUES (?, ?, ?)",
                    metric_pred_rows,
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS metrics_history (metric TEXT, value REAL)"
            )
            conn.execute("DELETE FROM metrics_history")
            metric_rows: list[tuple[str, float | None]] = []
            all_metric_histories = {
                **self.metrics_history,
                **self.synergy_metrics_history,
            }
            for m, vals in all_metric_histories.items():
                if vals:
                    metric_rows.extend((m, float(v)) for v in vals)
                else:
                    metric_rows.append((m, None))
            if metric_rows:
                conn.executemany(
                    "INSERT INTO metrics_history (metric, value) VALUES (?, ?)",
                    metric_rows,
                )
            conn.execute("CREATE TABLE IF NOT EXISTS synergy_history (data TEXT)")
            conn.execute("DELETE FROM synergy_history")
            if self.synergy_history:
                conn.executemany(
                    "INSERT INTO synergy_history (data) VALUES (?)",
                    [(json.dumps(d),) for d in self.synergy_history],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenario_synergy (scenario TEXT, data TEXT)"
            )
            conn.execute("DELETE FROM scenario_synergy")
            scenario_rows = [
                (scen, json.dumps(d))
                for scen, lst in self.scenario_synergy.items()
                for d in lst
            ]
            if scenario_rows:
                conn.executemany(
                    "INSERT INTO scenario_synergy (scenario, data) VALUES (?, ?)",
                    scenario_rows,
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenario_deltas (scenario TEXT, delta REAL)"
            )
            conn.execute("DELETE FROM scenario_deltas")
            if self.scenario_roi_deltas:
                conn.executemany(
                    "INSERT INTO scenario_deltas (scenario, delta) VALUES (?, ?)",
                    [
                        (scen, float(delta))
                        for scen, delta in self.scenario_roi_deltas.items()
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenario_raroi (scenario TEXT, raroi REAL)"
            )
            conn.execute("DELETE FROM scenario_raroi")
            if self.scenario_raroi:
                conn.executemany(
                    "INSERT INTO scenario_raroi (scenario, raroi) VALUES (?, ?)",
                    [
                        (scen, float(val)) for scen, val in self.scenario_raroi.items()
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenario_raroi_delta (scenario TEXT, delta REAL)"
            )
            conn.execute("DELETE FROM scenario_raroi_delta")
            if self.scenario_raroi_delta:
                conn.executemany(
                    "INSERT INTO scenario_raroi_delta (scenario, delta) VALUES (?, ?)",
                    [
                        (scen, float(val))
                        for scen, val in self.scenario_raroi_delta.items()
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenario_metrics_delta (scenario TEXT, data TEXT)"
            )
            conn.execute("DELETE FROM scenario_metrics_delta")
            if self.scenario_metrics_delta:
                conn.executemany(
                    "INSERT INTO scenario_metrics_delta (scenario, data) VALUES (?, ?)",
                    [
                        (scen, json.dumps(data))
                        for scen, data in self.scenario_metrics_delta.items()
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scenario_synergy_delta (scenario TEXT, data TEXT)"
            )
            conn.execute("DELETE FROM scenario_synergy_delta")
            if self.scenario_synergy_delta:
                conn.executemany(
                    "INSERT INTO scenario_synergy_delta (scenario, data) VALUES (?, ?)",
                    [
                        (scen, json.dumps(data))
                        for scen, data in self.scenario_synergy_delta.items()
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS worst_scenario (scenario TEXT)"
            )
            conn.execute("DELETE FROM worst_scenario")
            worst_label, _ = self.biggest_drop()
            if worst_label:
                conn.execute(
                    "INSERT INTO worst_scenario (scenario) VALUES (?)",
                    (worst_label,),
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS drift_metrics (score REAL, flag INTEGER)"
            )
            conn.execute("DELETE FROM drift_metrics")
            if self.drift_scores:
                conn.executemany(
                    "INSERT INTO drift_metrics (score, flag) VALUES (?, ?)",
                    [
                        (float(s), int(f))
                        for s, f in zip(self.drift_scores, self.drift_flags)
                    ],
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS drift_summary (metric TEXT, value REAL)"
            )
            conn.execute("DELETE FROM drift_summary")
            if self.drift_metrics:
                conn.executemany(
                    "INSERT INTO drift_summary (metric, value) VALUES (?, ?)",
                    [
                        (str(k), float(v))
                        for k, v in self.drift_metrics.items()
                    ],
                )

        save_all_summaries(Path(path).parent)

    # ------------------------------------------------------------------
    def load_history(self, path: str) -> None:
        """Populate ``roi_history`` and ``module_deltas`` from ``path``."""
        if not os.path.exists(path):
            return
        if path.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self.roi_history = [float(x) for x in data]
                    self.raroi_history = [0.0] * len(self.roi_history)
                    self.confidence_history = [0.0] * len(self.roi_history)
                    self.module_deltas = {}
                    self.module_raroi = {}
                    self.module_entropy_deltas = {}
                    self._origin_db_deltas = {}
                    self.db_roi_metrics = {}
                    self.predicted_roi = []
                    self.actual_roi = []
                    self.category_history = []
                    self.drift_scores = []
                    self.drift_flags = []
                    self.metrics_history = {}
                    self.predicted_metrics = {}
                    self.actual_metrics = {}
                    self.synergy_history = []
                    self.scenario_synergy = {}
                    self.scenario_roi_deltas = {}
                    self.scenario_raroi = {}
                    self.scenario_raroi_delta = {}
                    self._worst_scenario = None
                else:
                    self.roi_history = [float(x) for x in data.get("roi_history", [])]
                    self.raroi_history = [float(x) for x in data.get("raroi_history", [])]
                    self.confidence_history = [
                        float(x) for x in data.get("confidence_history", [])
                    ]
                    if len(self.confidence_history) < len(self.roi_history):
                        self.confidence_history.extend(
                            [0.0] * (len(self.roi_history) - len(self.confidence_history))
                        )
                    elif len(self.confidence_history) > len(self.roi_history):
                        self.confidence_history = self.confidence_history[: len(self.roi_history)]
                    if len(self.raroi_history) < len(self.roi_history):
                        self.raroi_history.extend(
                            [0.0] * (len(self.roi_history) - len(self.raroi_history))
                        )
                    elif len(self.raroi_history) > len(self.roi_history):
                        self.raroi_history = self.raroi_history[: len(self.roi_history)]
                    self.module_deltas = {
                        str(m): [float(v) for v in vals]
                        for m, vals in data.get("module_deltas", {}).items()
                    }
                    self.module_raroi = {
                        str(m): [float(v) for v in vals]
                        for m, vals in data.get("module_raroi", {}).items()
                    }
                    self.module_entropy_deltas = {
                        str(m): [float(v) for v in vals]
                        for m, vals in data.get("module_entropy_deltas", {}).items()
                    }
                    self._origin_db_deltas = {
                        str(m): [float(v) for v in vals]
                        for m, vals in data.get("origin_db_deltas", {}).items()
                    }
                    self.db_roi_metrics = {
                        str(db): {
                            "win_rate": float(stats.get("win_rate", 0.0)),
                            "regret_rate": float(stats.get("regret_rate", 0.0)),
                            "roi": float(stats.get("roi", 0.0)),
                        }
                        for db, stats in data.get("db_roi_metrics", {}).items()
                        if isinstance(stats, dict)
                    }
                    self.predicted_roi = [
                        float(x) for x in data.get("predicted_roi", [])
                    ]
                    self.actual_roi = [float(x) for x in data.get("actual_roi", [])]
                    self.category_history = [
                        str(x) for x in data.get("category_history", [])
                    ]
                    self.drift_scores = [
                        float(x) for x in data.get("drift_scores", [])
                    ]
                    self.drift_flags = [
                        bool(x) for x in data.get("drift_flags", [])
                    ]
                    self.metrics_history = {
                        str(m): [float(v) for v in vals]
                        for m, vals in data.get("metrics_history", {}).items()
                    }
                    self.synergy_metrics_history = {
                        str(m): [float(v) for v in vals]
                        for m, vals in data.get("synergy_metrics_history", {}).items()
                    }
                    self.synergy_history = [
                        {
                            str(k): float(v)
                            for k, v in entry.items()
                            if isinstance(entry, dict)
                        }
                        for entry in data.get("synergy_history", [])
                        if isinstance(entry, dict)
                    ]
                    self.workflow_mae_history = defaultdict(
                        list,
                        {
                            str(k): [float(v) for v in vals]
                            for k, vals in data.get("workflow_mae_history", {}).items()
                        },
                    )
                    self.workflow_variance_history = defaultdict(
                        list,
                        {
                            str(k): [float(v) for v in vals]
                            for k, vals in data.get("workflow_variance_history", {}).items()
                        },
                    )
                    self.workflow_confidence_history = defaultdict(
                        list,
                        {
                            str(k): [float(v) for v in vals]
                            for k, vals in data.get("workflow_confidence_history", {}).items()
                        },
                    )
                    self.scenario_synergy = {
                        str(n): [
                            {
                                str(k): float(v)
                                for k, v in entry.items()
                                if isinstance(entry, dict)
                            }
                            for entry in lst or []
                            if isinstance(entry, dict)
                        ]
                        for n, lst in data.get("scenario_synergy", {}).items()
                        if isinstance(lst, list)
                    }
                    self.scenario_roi_deltas = {
                        str(k): float(v)
                        for k, v in data.get("scenario_roi_deltas", {}).items()
                    }
                    self.scenario_raroi = {
                        str(k): float(v)
                        for k, v in data.get("scenario_raroi", {}).items()
                    }
                    self.scenario_raroi_delta = {
                        str(k): float(v)
                        for k, v in data.get("scenario_raroi_delta", {}).items()
                    }
                    self.scenario_metrics_delta = {
                        str(n): {
                            str(k): float(v)
                            for k, v in (entry or {}).items()
                        }
                        for n, entry in data.get("scenario_metrics_delta", {}).items()
                    }
                    self.scenario_synergy_delta = {
                        str(n): {
                            str(k): float(v)
                            for k, v in (entry or {}).items()
                        }
                        for n, entry in data.get("scenario_synergy_delta", {}).items()
                    }
                    worst = data.get("worst_scenario")
                    self._worst_scenario = str(worst) if worst is not None else None
                    if not self.synergy_metrics_history:
                        for m, vals in data.get("metrics_history", {}).items():
                            if str(m).startswith("synergy_"):
                                self.synergy_metrics_history[str(m)] = [
                                    float(v) for v in vals
                                ]
                    metric_preds = data.get("metric_predictions")
                    if isinstance(metric_preds, dict):
                        self.predicted_metrics = {}
                        self.actual_metrics = {}
                        for m, pairs in metric_preds.items():
                            p_list, a_list = [], []
                            for pair in pairs or []:
                                try:
                                    p, a = pair
                                    p_list.append(float(p))
                                    a_list.append(float(a))
                                except Exception:
                                    continue
                            if p_list:
                                self.predicted_metrics[str(m)] = p_list
                            if a_list:
                                self.actual_metrics[str(m)] = a_list
                    else:
                        self.predicted_metrics = {
                            str(m): [float(v) for v in vals]
                            for m, vals in data.get("predicted_metrics", {}).items()
                        }
                        self.actual_metrics = {
                            str(m): [float(v) for v in vals]
                            for m, vals in data.get("actual_metrics", {}).items()
                        }
                    n = len(self.roi_history)
                    self.metrics_history.setdefault("recovery_time", [0.0] * n)
                    while len(self.metrics_history["recovery_time"]) < n:
                        self.metrics_history["recovery_time"].append(0.0)
                    for key in (
                        "synergy_discrepancy_count",
                        "synergy_gpu_usage",
                        "synergy_cpu_usage",
                        "synergy_memory_usage",
                        "synergy_revenue",
                        "synergy_long_term_lucrativity",
                    ):
                        self.synergy_metrics_history.setdefault(key, [])
                    self.drift_metrics = {
                        str(k): float(v)
                        for k, v in data.get("drift_metrics", {}).items()
                    }
                    ta = data.get("truth_adapter")
                    if self.truth_adapter is not None and isinstance(ta, dict):
                        try:
                            self.truth_adapter.metadata.update(ta)
                        except Exception:
                            pass
            except Exception:
                self.roi_history = []
                self.module_deltas = {}
                self.module_entropy_deltas = {}
                self.metrics_history = {}
                self.scenario_synergy = {}
                self.scenario_roi_deltas = {}
                self.scenario_raroi = {}
                self.scenario_raroi_delta = {}
                self.scenario_metrics_delta = {}
                self.scenario_synergy_delta = {}
                self._worst_scenario = None
            return
        try:
            with router.get_connection("roi_history") as conn:
                try:
                    rows = conn.execute(
                        "SELECT delta, confidence, raroi, mae, roi_variance, final_score FROM roi_history ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    try:
                        rows = conn.execute(
                            "SELECT delta, confidence, raroi FROM roi_history ORDER BY rowid"
                        ).fetchall()
                        rows = [(*r, 0.0, 0.0, 0.0) for r in rows]
                    except Exception:
                        try:
                            rows = conn.execute(
                                "SELECT delta, confidence FROM roi_history ORDER BY rowid"
                            ).fetchall()
                            rows = [(r[0], r[1], 0.0, 0.0, 0.0, 0.0) for r in rows]
                        except Exception:
                            rows = conn.execute(
                                "SELECT delta FROM roi_history ORDER BY rowid"
                            ).fetchall()
                            rows = [(r[0], 0.0, 0.0, 0.0, 0.0, 0.0) for r in rows]
                try:
                    mod_rows = conn.execute(
                        "SELECT module, delta, raroi FROM module_deltas ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    mod_rows = conn.execute(
                        "SELECT module, delta FROM module_deltas ORDER BY rowid"
                    ).fetchall()
                    mod_rows = [(r[0], r[1], 0.0) for r in mod_rows]
                try:
                    ent_rows = conn.execute(
                        "SELECT module, delta FROM module_entropy_deltas ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    ent_rows = []
                try:
                    db_rows = conn.execute(
                        "SELECT origin_db, delta FROM db_deltas ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    db_rows = []
                try:
                    roi_metric_rows = conn.execute(
                        "SELECT origin_db, win_rate, regret_rate, roi FROM db_roi_metrics ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    roi_metric_rows = []
                pred_rows = conn.execute(
                    "SELECT pred, actual FROM predictions ORDER BY rowid"
                ).fetchall()
                try:
                    metric_rows = conn.execute(
                        "SELECT metric, value FROM metrics_history ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    metric_rows = []
                try:
                    metric_pred_rows = conn.execute(
                        "SELECT metric, pred, actual FROM metric_predictions ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    metric_pred_rows = []
                try:
                    synergy_rows = conn.execute(
                        "SELECT data FROM synergy_history ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    synergy_rows = []
                try:
                    scenario_rows = conn.execute(
                        "SELECT scenario, data FROM scenario_synergy ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    scenario_rows = []
                try:
                    scen_delta_rows = conn.execute(
                        "SELECT scenario, delta FROM scenario_deltas ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    scen_delta_rows = []
                try:
                    scen_raroi_rows = conn.execute(
                        "SELECT scenario, raroi FROM scenario_raroi ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    scen_raroi_rows = []
                try:
                    scen_raroi_delta_rows = conn.execute(
                        "SELECT scenario, delta FROM scenario_raroi_delta ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    scen_raroi_delta_rows = []
                try:
                    scen_metric_rows = conn.execute(
                        "SELECT scenario, data FROM scenario_metrics_delta ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    scen_metric_rows = []
                try:
                    scen_syn_rows = conn.execute(
                        "SELECT scenario, data FROM scenario_synergy_delta ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    scen_syn_rows = []
                try:
                    worst_row = conn.execute(
                        "SELECT scenario FROM worst_scenario ORDER BY rowid LIMIT 1"
                    ).fetchone()
                except Exception:
                    worst_row = None
                try:
                    cat_rows = conn.execute(
                        "SELECT category FROM categories ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    cat_rows = []
                try:
                    drift_rows = conn.execute(
                        "SELECT score, flag FROM drift_metrics ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    drift_rows = []
                try:
                    drift_sum_rows = conn.execute(
                        "SELECT metric, value FROM drift_summary ORDER BY rowid"
                    ).fetchall()
                except Exception:
                    drift_sum_rows = []
        except Exception:
            self.roi_history = []
            self.module_deltas = {}
            self.module_entropy_deltas = {}
            self.predicted_roi = []
            self.actual_roi = []
            self.metrics_history = {}
            self.predicted_metrics = {}
            self.actual_metrics = {}
            self.scenario_synergy = {}
            self.scenario_roi_deltas = {}
            self.scenario_raroi = {}
            self.scenario_raroi_delta = {}
            self._worst_scenario = None
            self.category_history = []
            self.drift_scores = []
            self.drift_flags = []
            self.drift_metrics = {}
            self.db_roi_metrics = {}
            return
        self.roi_history = [float(r[0]) for r in rows]
        self.confidence_history = [float(r[1]) for r in rows]
        self.raroi_history = [float(r[2]) for r in rows]
        self.mae_history = [float(r[3]) for r in rows]
        self.variance_history = [float(r[4]) for r in rows]
        self.final_roi_history["_global"] = [float(r[5]) for r in rows]
        self.module_deltas = {}
        self.module_raroi = {}
        for mod, delta, raroi in mod_rows:
            self.module_deltas.setdefault(str(mod), []).append(float(delta))
            self.module_raroi.setdefault(str(mod), []).append(float(raroi))
        self.module_entropy_deltas = {}
        for mod, delta in ent_rows:
            self.module_entropy_deltas.setdefault(str(mod), []).append(float(delta))
        self._origin_db_deltas = {}
        for db, delta in db_rows:
            self._origin_db_deltas.setdefault(str(db), []).append(float(delta))
        self.db_roi_metrics = {}
        for db, win, regret, roi in roi_metric_rows:
            self.db_roi_metrics[str(db)] = {
                "win_rate": float(win),
                "regret_rate": float(regret),
                "roi": float(roi),
            }
        self.predicted_roi = [float(r[0]) for r in pred_rows]
        self.actual_roi = [float(r[1]) for r in pred_rows]
        self.category_history = [str(r[0]) for r in cat_rows]
        self.drift_scores = [float(r[0]) for r in drift_rows]
        self.drift_flags = [bool(r[1]) for r in drift_rows]
        self.drift_metrics = {str(m): float(v) for m, v in drift_sum_rows}
        self.metrics_history = {}
        self.synergy_metrics_history = {}
        for name, val in metric_rows:
            target = (
                self.synergy_metrics_history
                if str(name).startswith("synergy_")
                else self.metrics_history
            )
            if val is None:
                target.setdefault(str(name), [])
            else:
                target.setdefault(str(name), []).append(float(val))
        for name, vals in self.synergy_metrics_history.items():
            self.metrics_history.setdefault(name, list(vals))
        self.predicted_metrics = {}
        self.actual_metrics = {}
        for m, pred, act in metric_pred_rows:
            self.predicted_metrics.setdefault(str(m), []).append(float(pred))
            self.actual_metrics.setdefault(str(m), []).append(float(act))
        self.synergy_history = []
        for (json_data,) in synergy_rows:
            try:
                entry = json.loads(str(json_data))
                if isinstance(entry, dict):
                    self.synergy_history.append(
                        {str(k): float(v) for k, v in entry.items()}
                    )
            except Exception:
                continue
        self.scenario_synergy = {}
        for scen, json_data in scenario_rows:
            try:
                entry = json.loads(str(json_data))
                if isinstance(entry, dict):
                    self.scenario_synergy.setdefault(str(scen), []).append(
                        {str(k): float(v) for k, v in entry.items()}
                    )
            except Exception:
                continue
        self.scenario_roi_deltas = {
            str(scen): float(delta) for scen, delta in scen_delta_rows
        }
        self.scenario_raroi = {
            str(scen): float(val) for scen, val in scen_raroi_rows
        }
        self.scenario_raroi_delta = {
            str(scen): float(val) for scen, val in scen_raroi_delta_rows
        }
        self.scenario_metrics_delta = {}
        for scen, json_data in scen_metric_rows:
            try:
                entry = json.loads(str(json_data))
                if isinstance(entry, dict):
                    self.scenario_metrics_delta[str(scen)] = {
                        str(k): float(v) for k, v in entry.items()
                    }
            except Exception:
                continue
        self.scenario_synergy_delta = {}
        for scen, json_data in scen_syn_rows:
            try:
                entry = json.loads(str(json_data))
                if isinstance(entry, dict):
                    self.scenario_synergy_delta[str(scen)] = {
                        str(k): float(v) for k, v in entry.items()
                    }
            except Exception:
                continue
        self._worst_scenario = (
            str(worst_row[0]) if worst_row and worst_row[0] is not None else None
        )
        n = len(self.roi_history)
        self.metrics_history.setdefault("recovery_time", [0.0] * n)
        while len(self.metrics_history["recovery_time"]) < n:
            self.metrics_history["recovery_time"].append(0.0)
        for key in (
            "synergy_discrepancy_count",
            "synergy_gpu_usage",
            "synergy_cpu_usage",
            "synergy_memory_usage",
            "synergy_long_term_lucrativity",
            "synergy_reliability",
            "synergy_maintainability",
            "synergy_throughput",
        ):
            self.synergy_metrics_history.setdefault(key, [])

    # ------------------------------------------------------------------
    def impact_severity(self, workflow_type: str) -> float:
        """Return impact severity for ``workflow_type``."""

        return float(get_impact_severity(workflow_type))

    # ------------------------------------------------------------------
    @property
    def impact_severity_map(self) -> Mapping[str, float]:
        """Expose the loaded impact severity mapping."""

        return load_impact_severity_map()

    # ------------------------------------------------------------------
    def _safety_factor(self, metrics: Mapping[str, float]) -> float:
        """Derive safety factor from security/alignment metrics."""

        safety_keys = [
            "safety_rating",
            "security_score",
            "synergy_safety_rating",
            "synergy_security_score",
        ]
        vals = [
            float(metrics.get(k, 0.0))
            for k in safety_keys
            if metrics.get(k) is not None
        ]
        if not vals:
            for k in safety_keys:
                hist = self.metrics_history.get(k) or self.synergy_metrics_history.get(k)
                if hist:
                    vals.append(float(hist[-1]))
        base = float(np.mean(vals)) if vals else 1.0
        base = max(0.0, min(1.0, base))
        fail_keys = [
            "hostile_failures",
            "misuse_failures",
            "synergy_hostile_failures",
            "synergy_misuse_failures",
        ]
        failures = sum(float(metrics.get(k, 0.0)) for k in fail_keys)
        penalty = 1.0 / (1.0 + failures)

        # Apply penalties for failing critical suites where metrics report
        # ``<suite>_failures`` counts.  Each failure multiplies the safety
        # factor by the configured suite penalty.
        for name, mult in CRITICAL_TEST_PENALTIES.items():
            count = float(metrics.get(f"{name}_failures", 0.0))
            if count:
                penalty *= mult ** count

        return max(0.0, min(1.0, base * penalty))

    # ------------------------------------------------------------------
    def calculate_raroi(
        self,
        base_roi: float,
        *,
        workflow_type: str | None = None,
        rollback_prob: float | None = None,
        impact_severity: float | None = None,
        metrics: Mapping[str, float] | None = None,
        failing_tests: Iterable[str] | Mapping[str, bool] | None = None,
    ) -> tuple[float, float, list[tuple[str, str]]]:
        """Return ``(base_roi, risk_adjusted_roi, bottlenecks)`` for ``workflow_type``.

        The method estimates the catastrophic risk of continuing a workflow by
        combining rollback probability, workflow impact severity and recent
        stability metrics. Safety is further reduced when critical suites such
        as ``security`` or ``alignment`` fail. When ``rollback_prob`` is not
        provided, ``metrics`` supplies runtime information for estimating
        rollback probability via :func:`_estimate_rollback_probability`.

        ``failing_tests`` may explicitly list failing critical suites or map
        suite names to boolean pass/fail flags. When omitted, failing suites are
        looked up via :func:`self_test_service.get_failed_critical_tests`.
        If the resulting RAROI dips below ``raroi_borderline_threshold`` the
        function invokes :func:`propose_fix` to return bottleneck suggestions
        alongside the score.
        """

        recent = self.roi_history[-self.window :]
        instability = float(np.std(recent)) if recent else 0.0

        metrics_map: dict[str, float] = dict(metrics or {})
        metrics_map.setdefault("instability", instability)
        if hasattr(self, "_last_errors_per_minute"):
            metrics_map.setdefault(
                "errors_per_minute", float(self._last_errors_per_minute)
            )

        if rollback_prob is None:
            rollback_prob = _estimate_rollback_probability(metrics_map)
        rollback_prob = max(0.0, min(1.0, float(rollback_prob)))

        if impact_severity is None:
            wf = workflow_type or "standard"
            impact_severity = float(get_impact_severity(wf))
        else:
            impact_severity = float(impact_severity)

        impact_severity = max(0.0, min(1.0, impact_severity))

        catastrophic_risk = rollback_prob * impact_severity

        stability_factor = max(0.0, 1.0 - instability)

        safety_metrics: dict[str, float] = dict(metrics_map)
        if failing_tests is not None:
            if isinstance(failing_tests, Mapping):
                failures: Iterable[str] = [
                    str(name).lower()
                    for name, passed in failing_tests.items()
                    if not passed
                ]
            else:
                failures = [str(f).lower() for f in failing_tests]
        else:
            sts_module = _get_self_test_service()
            if sts_module is not None:
                try:
                    failures = [
                        str(f).lower() for f in sts_module.get_failed_critical_tests()
                    ]
                except Exception:
                    failures = []
            else:
                failures = []
        for name in failures:
            key = f"{name}_failures"
            safety_metrics[key] = safety_metrics.get(key, 0.0) + 1.0
        safety_factor = self._safety_factor(safety_metrics)

        raroi = float(
            base_roi * (1.0 - catastrophic_risk) * stability_factor * safety_factor
        )

        bottlenecks: list[tuple[str, str]] = []
        if raroi < self.raroi_borderline_threshold:
            try:
                profile = workflow_type or "standard"
                bottlenecks = propose_fix(metrics_map, profile)
            except Exception:
                bottlenecks = []

        self.last_raroi = raroi
        return float(base_roi), raroi, bottlenecks

    # ------------------------------------------------------------------
    def plot_history(self, output_path: str) -> None:
        """Plot recorded ROI deltas and fitted regression curve."""

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - matplotlib may be missing
            return

        if not self.roi_history:
            plt.figure()
            plt.savefig(output_path)
            plt.close()
            return

        x = np.arange(len(self.roi_history))
        y = np.array(self.roi_history)

        vertex, preds = self._regression()

        plt.figure()
        plt.plot(x, y, "o", label="delta")
        if preds:
            plt.plot(x, preds, label="fit")
        if vertex is not None:
            plt.axvline(vertex, color="red", linestyle="--", label="vertex")
        plt.xlabel("iteration")
        plt.ylabel("ROI delta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # ------------------------------------------------------------------
    def entropy_delta_history(self, module_name: str) -> List[float]:
        """Return recorded entropy delta ratios for ``module_name``."""

        return list(self.module_entropy_deltas.get(module_name, []))

    # ------------------------------------------------------------------
    def entropy_plateau(self, threshold: float, consecutive: int) -> List[str]:
        """Return modules whose entropy deltas stay below ``threshold``."""

        flags: List[str] = []
        thr = float(threshold)
        for mod, vals in self.module_entropy_deltas.items():
            if len(vals) < consecutive:
                continue
            window = vals[-consecutive:]
            if all(abs(v) <= thr for v in window):
                flags.append(mod)
        return flags

    # ------------------------------------------------------------------
    def rankings(self) -> List[Tuple[str, float, float]]:
        """Return modules sorted by cumulative risk-adjusted ROI.

        Each tuple contains ``(module, raroi_total, roi_total)``.
        """
        modules = set(self.module_deltas) | set(self.module_raroi)
        rows: List[Tuple[str, float, float]] = []
        for mod in modules:
            raroi_total = sum(self.module_raroi.get(mod, []))
            roi_total = sum(self.module_deltas.get(mod, []))
            rows.append((mod, raroi_total, roi_total))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    def cluster_rankings(self) -> List[Tuple[int, float, float]]:
        """Return clusters sorted by cumulative risk-adjusted ROI.

        Each tuple contains ``(cluster_id, raroi_total, roi_total)``.
        """
        clusters = set(self.cluster_deltas) | set(self.cluster_raroi)
        rows: List[Tuple[int, float, float]] = []
        for cid in clusters:
            raroi_total = sum(self.cluster_raroi.get(cid, []))
            roi_total = sum(self.cluster_deltas.get(cid, []))
            rows.append((cid, raroi_total, roi_total))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    def origin_db_deltas(self) -> Dict[str, float]:
        """Return latest ROI delta per origin database."""

        return {
            db: vals[-1]
            for db, vals in self._origin_db_deltas.items()
            if vals
        }

    # Expose the raw history for internal consumers and tests.
    @property
    def origin_db_delta_history(self) -> Dict[str, List[float]]:
        return self._origin_db_deltas

    # ------------------------------------------------------------------
    def update_db_metrics(
        self, metrics: Dict[str, Dict[str, float]], *, sqlite_path: str | None = None
    ) -> None:
        """Consume aggregated retrieval metrics grouped by ``origin_db``.

        Parameters
        ----------
        metrics:
            Mapping of origin database to a metrics dictionary containing
            ``roi`` (contribution), ``win_rate`` and ``regret_rate`` values.
        sqlite_path:
            Optional SQLite file where metrics are appended for historical
            analysis.
        """

        for origin, stats in metrics.items():
            roi = float(stats.get("roi", 0.0))
            win_rate = float(stats.get("win_rate", 0.0))
            regret_rate = float(stats.get("regret_rate", 0.0))
            # Always record deltas so neutral performance is captured and
            # ranking weights reflect recent history even when ROI is zero.
            self._origin_db_deltas.setdefault(origin, []).append(roi)
            self.db_roi_metrics[origin] = {
                "win_rate": win_rate,
                "regret_rate": regret_rate,
                "roi": roi,
            }
            if _DB_ROI_GAUGE is not None and self._origin_db_deltas.get(origin):
                avg = sum(self._origin_db_deltas[origin]) / len(
                    self._origin_db_deltas[origin]
                )
                _DB_ROI_GAUGE.labels(origin_db=origin).set(avg)

        if sqlite_path:
            with router.get_connection("db_roi_metrics") as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS db_roi_metrics (
                        origin_db TEXT,
                        win_rate REAL,
                        regret_rate REAL,
                        roi REAL
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO db_roi_metrics (origin_db, win_rate, regret_rate, roi) VALUES (?,?,?,?)",
                    [
                        (
                            db,
                            float(m.get("win_rate", 0.0)),
                            float(m.get("regret_rate", 0.0)),
                            float(m.get("roi", 0.0)),
                        )
                        for db, m in metrics.items()
                    ],
                )
                conn.commit()

    # ------------------------------------------------------------------
    def ingest_vector_metrics_db(
        self, vec_db: "VectorMetricsDB", *, sqlite_path: str | None = None
    ) -> Dict[str, Dict[str, float]]:
        """Load aggregated retrieval metrics from a :class:`VectorMetricsDB`.

        Parameters
        ----------
        vec_db:
            Instance of :class:`VectorMetricsDB` providing raw retrieval
            records. Metrics are aggregated by ``origin_db`` and passed to
            :meth:`update_db_metrics`.
        sqlite_path:
            Optional SQLite file where the aggregated metrics are stored.
        """

        cur = vec_db.conn.execute(
            """
            SELECT db, AVG(win) AS win_rate, AVG(regret) AS regret_rate,
                   COALESCE(SUM(contribution),0) AS roi
              FROM vector_metrics
             WHERE event_type='retrieval'
          GROUP BY db
            """
        )
        metrics = {
            str(db or ""): {
                "win_rate": float(win or 0.0),
                "regret_rate": float(regret or 0.0),
                "roi": float(roi or 0.0),
            }
            for db, win, regret, roi in cur.fetchall()
        }
        self.update_db_metrics(metrics, sqlite_path=sqlite_path)
        return metrics

    # ------------------------------------------------------------------
    def roi_by_origin_db(self) -> Dict[str, float]:
        """Return average ROI contribution per ``origin_db``."""

        return {
            db: (sum(vals) / len(vals))
            for db, vals in self._origin_db_deltas.items()
            if vals
        }

    # ------------------------------------------------------------------
    def db_roi_report(self) -> List[Dict[str, float]]:
        """Return ROI contribution, win-rate and regret-rate per database."""

        averages = self.roi_by_origin_db()
        report: List[Dict[str, float]] = []
        for db, avg in averages.items():
            stats = self.db_roi_metrics.get(db, {})
            report.append(
                {
                    "origin_db": db,
                    "avg_roi": avg,
                    "win_rate": float(stats.get("win_rate", 0.0)),
                    "regret_rate": float(stats.get("regret_rate", 0.0)),
                }
            )
        report.sort(key=lambda r: (-r["win_rate"], r["regret_rate"]))
        return report

    # ------------------------------------------------------------------
    def best_db_performance(self) -> Dict[str, Dict[str, float]]:
        """Return databases with highest win-rate and lowest regret.

        The returned mapping contains two entries: ``"highest_win_rate"`` and
        ``"lowest_regret"``.  Each entry provides the originating database name
        along with its win-rate, regret-rate and cumulative ROI contribution.
        When no metrics have been recorded an empty dictionary is returned.
        """

        if not self.db_roi_metrics:
            return {}
        best_win = max(
            self.db_roi_metrics.items(),
            key=lambda kv: float(kv[1].get("win_rate", 0.0)),
        )
        lowest_regret = min(
            self.db_roi_metrics.items(),
            key=lambda kv: float(kv[1].get("regret_rate", float("inf"))),
        )
        return {
            "highest_win_rate": {
                "origin_db": best_win[0],
                "win_rate": float(best_win[1].get("win_rate", 0.0)),
                "regret_rate": float(best_win[1].get("regret_rate", 0.0)),
                "roi": float(best_win[1].get("roi", 0.0)),
            },
            "lowest_regret": {
                "origin_db": lowest_regret[0],
                "win_rate": float(lowest_regret[1].get("win_rate", 0.0)),
                "regret_rate": float(lowest_regret[1].get("regret_rate", 0.0)),
                "roi": float(lowest_regret[1].get("roi", 0.0)),
            },
        }

    # ------------------------------------------------------------------
    def export_origin_db_roi_csv(self, path: str) -> None:
        """Write ROI contribution report per ``origin_db`` to ``path`` as CSV."""

        rows = self.db_roi_report()
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "origin_db",
                "avg_roi",
                "win_rate",
                "regret_rate",
                "count",
            ])
            for row in rows:
                writer.writerow(
                    [
                        row["origin_db"],
                        row["avg_roi"],
                        row["win_rate"],
                        row["regret_rate"],
                        len(self._origin_db_deltas.get(row["origin_db"], [])),
                    ]
                )

    # ------------------------------------------------------------------
    def retrieval_bias(self) -> Dict[str, float]:
        """Return multiplicative bias weights for databases based on ROI."""

        averages = self.roi_by_origin_db()
        if not averages:
            return {}
        max_abs = max(abs(v) for v in averages.values()) or 1.0
        bias: Dict[str, float] = {}
        for db, val in averages.items():
            weight = 1.0 + val / max_abs
            if weight < 0.1:
                weight = 0.1
            bias[db] = weight
        return bias


def load_workflow_module_deltas(
    workflow_id: str, run_id: str | None = None, db_path: str = "roi_results.db"
) -> Dict[str, float]:
    """Return aggregated ROI deltas per module for a workflow run."""

    def _conn_db_path(conn: sqlite3.Connection) -> str:
        return conn.execute("PRAGMA database_list").fetchone()[2]

    router: DBRouter
    if GLOBAL_ROUTER is not None and _conn_db_path(GLOBAL_ROUTER.local_conn) == str(db_path):
        router = GLOBAL_ROUTER
    else:
        router = DBRouter("workflow_results", str(db_path), str(db_path))
    conn = router.get_connection("workflow_module_deltas")
    cur = conn.cursor()
    try:
        if run_id is not None:
            cur.execute(
                "SELECT module, roi_delta FROM workflow_module_deltas WHERE workflow_id=? AND run_id=?",
                (workflow_id, run_id),
            )
        else:
            cur.execute(
                "SELECT module, SUM(roi_delta) FROM workflow_module_deltas WHERE workflow_id=? GROUP BY module",
                (workflow_id,),
            )
        rows = cur.fetchall()
    finally:
        cur.close()
        if router is not GLOBAL_ROUTER:
            router.close()
    return {str(m): float(d) for m, d in rows}


def apply_workflow_module_deltas(
    tracker: "ROITracker",
    workflow_id: str,
    run_id: str | None = None,
    db_path: str = "roi_results.db",
) -> Dict[str, float]:
    """Load module deltas and append them to ``tracker`` for refinement."""

    deltas = load_workflow_module_deltas(workflow_id, run_id, db_path)
    for mod, delta in deltas.items():
        tracker.module_deltas.setdefault(str(mod), []).append(float(delta))
    return deltas


def cli(argv: List[str] | None = None) -> None:
    """Command line interface for ``ROITracker`` utilities."""

    import argparse

    parser = argparse.ArgumentParser(description="ROITracker utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plot = sub.add_parser("plot", help="plot ROI history")
    p_plot.add_argument("history", help="history file (json or sqlite)")
    p_plot.add_argument("output", help="output image path")

    p_forecast = sub.add_parser("forecast", help="predict next ROI delta")
    p_forecast.add_argument("history", help="history file (json or sqlite)")

    p_rank = sub.add_parser("rank", help="show module rankings")
    p_rank.add_argument("history", help="history file (json or sqlite)")

    p_rel = sub.add_parser("reliability", help="show prediction MAE")
    p_rel.add_argument("history", help="history file (json or sqlite)")
    p_rel.add_argument("--window", type=int, default=None, help="rolling window")
    p_rel.add_argument("--metric", help="metric name (defaults to ROI)")

    p_pred = sub.add_parser(
        "predict-metric", help="predict metric via prediction manager"
    )
    p_pred.add_argument("history", help="history file (json or sqlite)")
    p_pred.add_argument("metric", help="metric name")
    p_pred.add_argument(
        "--actual", type=float, default=None, help="optional actual value"
    )

    args = parser.parse_args(argv)

    tracker = ROITracker()
    if args.cmd == "plot":
        tracker.load_history(args.history)
        tracker.plot_history(args.output)
    elif args.cmd == "forecast":
        tracker.load_history(args.history)
        pred, (lo, hi) = tracker.forecast()
        print(f"Predicted ROI: {pred:.3f} (CI {lo:.3f} - {hi:.3f})")
    elif args.cmd == "rank":
        tracker.load_history(args.history)
        for mod, raroi, roi in tracker.rankings():
            print(f"{mod} {raroi:.3f} (roi {roi:.3f})")
    elif args.cmd == "reliability":
        tracker.load_history(args.history)
        if args.metric:
            mae = tracker.rolling_mae_metric(args.metric, args.window)
            print(f"{args.metric} MAE: {mae:.3f}")
        else:
            roi_mae = tracker.rolling_mae(args.window)
            print(f"ROI MAE: {roi_mae:.3f}")
            for name in sorted(tracker.metrics_history):
                m_mae = tracker.rolling_mae_metric(name, args.window)
                print(f"{name} MAE: {m_mae:.3f}")
    elif args.cmd == "predict-metric":
        tracker.load_history(args.history)
        if args.metric == "synergy_profitability":
            val = tracker.predict_synergy_profitability()
            print(f"Predicted {args.metric}: {val:.3f}")
        elif args.metric == "synergy_revenue":
            val = tracker.predict_synergy_revenue()
            print(f"Predicted {args.metric}: {val:.3f}")
        elif args.metric == "synergy_projected_lucrativity":
            val = tracker.predict_synergy_projected_lucrativity()
            print(f"Predicted {args.metric}: {val:.3f}")
        else:
            from .prediction_manager_bot import PredictionManager

            manager = PredictionManager()
            val = tracker.predict_metric_with_manager(
                manager, args.metric, actual=args.actual
            )
            print(f"Predicted {args.metric}: {val:.3f}")


def main(argv: List[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
