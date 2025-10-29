from __future__ import annotations

"""Utilities for visualising evaluation results."""

import json
import logging
import queue
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List
import types
from .relevancy_radar import flagged_modules

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

# optional CLI support
import argparse

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional

    class _SimpleDataFrame(list):
        """Very small pandas.DataFrame replacement used when pandas is absent."""

        def __init__(self, records: List[Dict[str, Any]]):
            super().__init__(records)
            self.columns = list(records[0].keys()) if records else []

        def drop_duplicates(self, *, subset: List[str] | None = None):
            subset = subset or self.columns
            seen = set()
            dedup: List[Dict[str, Any]] = []
            for row in self:
                key = tuple(row.get(col) for col in subset)
                if key not in seen:
                    seen.add(key)
                    dedup.append(row)
            return _SimpleDataFrame(dedup)

        def to_dict(self, orient: str):  # pragma: no cover - minimal
            if orient != "records":
                raise ValueError("only orient='records' supported")
            return list(self)

    pd = types.SimpleNamespace(DataFrame=_SimpleDataFrame)

logger = logging.getLogger(__name__)


class _FallbackEvaluationManager:
    """Minimal replacement used when :mod:`evaluation_manager` is unavailable."""

    history: Dict[str, List[Dict[str, Any]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.history = {}

    def record(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        """Compatibility shim matching the real API."""


class _FallbackROITracker:
    """Lightweight ROI tracker stub used when the real implementation is missing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.predicted_roi: List[float] = []
        self.actual_roi: List[float] = []
        self.horizon_mae_history: List[Dict[str, Any]] = []
        self.drift_flags: List[Any] = []
        self.metrics_history: Dict[str, List[Any]] = {}
        self.workflow_predicted_roi: List[Any] = []
        self.needs_review: set[Any] = set()
        self.drift_metrics: Dict[str, Any] = {}

    def load_history(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        """Compatibility shim matching the real API."""

    def prediction_summary(self, window: int | None = None) -> Dict[str, Dict[str, Any]]:
        """Return empty structures compatible with the real tracker."""

        return {
            "mae": 0.0,
            "accuracy": 0.0,
            "class_counts": {"predicted": {}, "actual": {}},
            "confusion_matrix": {},
            "mae_trend": [],
            "accuracy_trend": [],
            "scenario_roi_deltas": {},
            "scenario_raroi": {},
            "scenario_raroi_delta": {},
            "scenario_metrics_delta": {},
            "scenario_synergy_delta": {},
            "worst_scenario": None,
            "workflow_confidence": {},
            "workflow_mae": {},
            "workflow_variance": {},
        }

    def classification_accuracy(self, window: int | None = None) -> float:
        """Return neutral accuracy for compatibility with the real tracker."""

        return 0.0


class _FallbackTelemetryBackend:
    """Telemetry stub returning empty history when backend is missing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        pass

    def fetch_history(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return []


def _load_dependency(name: str, module: str, attr: str, fallback: Any) -> Any:
    """Return attribute ``attr`` from ``module`` or ``fallback`` when import fails."""

    try:
        pkg = __package__ or "menace_sandbox"
        mod = __import__(f"{pkg}.{module}", fromlist=[attr])
        return getattr(mod, attr)
    except Exception as exc:  # pragma: no cover - exercised via tests
        logger.warning(
            "evaluation_dashboard: failed to import %s (%s); using fallback stub",
            name,
            exc,
        )
        return fallback


@lru_cache(maxsize=None)
def _get_evaluation_manager_cls() -> type[_FallbackEvaluationManager]:
    return _load_dependency(
        "EvaluationManager", "evaluation_manager", "EvaluationManager", _FallbackEvaluationManager
    )


@lru_cache(maxsize=None)
def _get_roi_tracker_cls() -> type[_FallbackROITracker]:
    return _load_dependency("ROITracker", "roi_tracker", "ROITracker", _FallbackROITracker)


@lru_cache(maxsize=None)
def _get_telemetry_backend_cls() -> type[_FallbackTelemetryBackend]:
    return _load_dependency(
        "TelemetryBackend", "telemetry_backend", "TelemetryBackend", _FallbackTelemetryBackend
    )


def _instantiate_dependency(
    name: str,
    loader: Any,
    fallback: type,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Instantiate dependency ``name`` using ``loader`` while handling failures."""

    try:
        cls = loader()
    except Exception as exc:  # pragma: no cover - exercised via tests
        logger.warning(
            "evaluation_dashboard: %s loader failed (%s); using fallback stub",
            name,
            exc,
        )
        cls = fallback
    try:
        return cls(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - exercised via tests
        logger.warning(
            "evaluation_dashboard: %s instantiation failed (%s); using fallback stub",
            name,
            exc,
        )
        return fallback(*args, **kwargs)
from .violation_logger import load_persisted_alignment_warnings
from .scope_utils import Scope

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .evaluation_manager import EvaluationManager as _EvaluationManagerType
    from .roi_tracker import ROITracker as _ROITrackerType
    from .telemetry_backend import TelemetryBackend as _TelemetryBackendType
else:  # Fallbacks used at runtime when optional deps are absent
    _EvaluationManagerType = _FallbackEvaluationManager
    _ROITrackerType = _FallbackROITracker
    _TelemetryBackendType = _FallbackTelemetryBackend

GOVERNANCE_LOG = resolve_path("sandbox_data") / "governance_outcomes.jsonl"


def append_governance_result(
    scorecard: Dict[str, Any],
    vetoes: List[str],
    forecast: Dict[str, Any] | None = None,
    reasons: List[str] | None = None,
) -> None:
    """Persist *scorecard* evaluation outcome for later review.

    Parameters
    ----------
    scorecard:
        The evaluation metrics collected for the workflow.
    vetoes:
        List of veto codes applied by governance rules.
    forecast, reasons:
        Optional foresight forecast data and reason codes to persist alongside
        the scorecard and vetoes.
    """

    rec = dict(scorecard)
    rec["vetoes"] = list(vetoes)
    if forecast is not None:
        rec["forecast"] = forecast
    if reasons is not None:
        rec["reasons"] = list(reasons)
    GOVERNANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GOVERNANCE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _build_tree(workflow_id: int) -> List[Dict[str, Any]]:
    """Helper returning the lineage tree for ``workflow_id``."""

    try:
        from .mutation_lineage import MutationLineage

        return MutationLineage().build_tree(workflow_id)
    except Exception:  # pragma: no cover - best effort fallback
        from .mutation_logger import build_lineage

        return build_lineage(workflow_id)


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    if name == "EvaluationManager":
        cls = _get_evaluation_manager_cls()
        globals()[name] = cls
        return cls
    if name == "ROITracker":
        cls = _get_roi_tracker_cls()
        globals()[name] = cls
        return cls
    if name == "TelemetryBackend":
        cls = _get_telemetry_backend_cls()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


class EvaluationDashboard:
    """Render :class:`EvaluationManager` history as data frames and weights."""

    def __init__(self, manager: _EvaluationManagerType) -> None:
        self.manager = manager
        self._lineage_lock = threading.Lock()
        self._lineage_trees: Dict[int, List[Dict[str, Any]]] = {}
        self._lineage_updates: queue.Queue[List[Dict[str, Any]]] = queue.Queue()

    def dataframe(self):
        if pd is None:
            return None
        records: List[Dict[str, Any]] = []
        for name, history in self.manager.history.items():
            for rec in history:
                r = dict(rec)
                r["engine"] = name
                records.append(r)
        if not records:
            return None
        return pd.DataFrame(records)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.manager.history, indent=2))

    def deployment_weights(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for name, hist in self.manager.history.items():
            if hist:
                totals[name] = sum(h.get("cv_score", 0.0) for h in hist)
                counts[name] = len(hist)
        if not totals:
            return {}
        max_avg = max(totals[n] / counts[n] for n in totals)
        return {n: (totals[n] / counts[n]) / max_avg for n in totals}

    # ------------------------------------------------------------------
    def alignment_warning_panel(
        self,
        limit: int = 50,
        min_severity: int | None = None,
        max_severity: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Return recent alignment warnings with optional severity filtering.

        Each record includes a ``patch_link`` or identifier allowing
        operators to inspect the related code patch.
        """

        return load_persisted_alignment_warnings(
            limit=limit, min_severity=min_severity, max_severity=max_severity
        )

    # ------------------------------------------------------------------
    def roi_prediction_panel(
        self, tracker: _ROITrackerType | None = None, window: int | None = None
    ) -> Dict[str, Any]:
        """Return ROI prediction stats including trend and confusion metrics."""
        tracker = _ensure_tracker(tracker)
        summary = tracker.prediction_summary(window)
        workflows: Dict[str, Any] = {}
        wf_conf = summary.get("workflow_confidence", {})
        wf_mae = summary.get("workflow_mae", {})
        wf_var = summary.get("workflow_variance", {})
        for wf in tracker.workflow_predicted_roi:
            wid = str(wf)
            workflows[wid] = {
                "confidence": wf_conf.get(wid, []),
                "mae": wf_mae.get(wid, []),
                "variance": wf_var.get(wid, []),
                "needs_review": wid in tracker.needs_review,
            }
        summary["workflows"] = workflows
        return summary

    # ------------------------------------------------------------------
    def roi_prediction_chart(
        self, tracker: _ROITrackerType | None = None, window: int | None = None
    ) -> Dict[str, Any]:
        """Return sequences for charting predicted vs actual ROI.

        Parameters
        ----------
        tracker:
            The :class:`ROITracker` instance holding prediction history.
        window:
            Optional number of recent samples to include. When omitted the
            entire history is returned.

        Returns
        -------
        dict
            Dictionary containing ``labels`` for the x-axis along with
            ``predicted`` and ``actual`` ROI sequences.
        """

        tracker = _ensure_tracker(tracker)
        total = len(tracker.predicted_roi)
        preds = tracker.predicted_roi[-window:] if window else tracker.predicted_roi
        acts = tracker.actual_roi[-len(preds):]
        start = total - len(preds)
        labels = list(range(start, start + len(preds)))
        return {"labels": labels, "predicted": preds, "actual": acts}

    # ------------------------------------------------------------------
    def roi_prediction_events_panel(
        self, tracker: _ROITrackerType | None = None, window: int | None = None
    ) -> Dict[str, Any]:
        """Summarise ``roi_prediction_events`` persisted by ``ROITracker``.

        Parameters
        ----------
        tracker:
            The :class:`ROITracker` instance holding event history.
        window:
            Optional number of recent events to consider when computing the
            growth class accuracy and drift flags. When omitted the entire
            history is used.

        Returns
        -------
        dict
            Dictionary containing ``mae_by_horizon`` for the most recent
            evaluation window, ``growth_class_accuracy`` over ``window`` samples
            and the list of recent ``drift_flags``.
        """

        tracker = _ensure_tracker(tracker)
        mae_by_horizon = (
            tracker.horizon_mae_history[-1] if tracker.horizon_mae_history else {}
        )
        drift_flags = tracker.drift_flags[-window:] if window else tracker.drift_flags
        predictor = getattr(tracker, "_adaptive_predictor", None)
        drift_metrics = getattr(predictor, "drift_metrics", {}) if predictor else {}
        if not drift_metrics:
            drift_metrics = getattr(tracker, "drift_metrics", {})
        summary = tracker.prediction_summary(window)
        wf_conf = summary.get("workflow_confidence", {})
        wf_mae = summary.get("workflow_mae", {})
        wf_var = summary.get("workflow_variance", {})
        workflows: Dict[str, Any] = {}
        for wf in tracker.workflow_predicted_roi:
            wid = str(wf)
            workflows[wid] = {
                "confidence": wf_conf.get(wid, []),
                "mae": wf_mae.get(wid, []),
                "variance": wf_var.get(wid, []),
                "needs_review": wid in tracker.needs_review,
            }
        return {
            "mae_by_horizon": mae_by_horizon,
            "growth_class_accuracy": tracker.classification_accuracy(window),
            "drift_flags": drift_flags,
            "growth_type_accuracy": drift_metrics.get("accuracy", 0.0),
            "drift_metrics": drift_metrics,
            "workflows": workflows,
        }

    # ------------------------------------------------------------------
    def readiness_chart(
        self,
        telemetry: TelemetryBackend,
        workflow_id: str | None = None,
        window: int | None = None,
        *,
        scope: Scope | str = "local",
    ) -> Dict[str, Any]:
        """Return readiness index values over time.

        Parameters
        ----------
        telemetry:
            Telemetry backend containing readiness snapshots.
        workflow_id:
            Optional workflow identifier to filter history.
        window:
            Optional number of most recent samples to return.

        Returns
        -------
        dict
            Dictionary with ``labels`` (timestamps) and ``readiness`` values.
        """

        history = telemetry.fetch_history(workflow_id, scope=scope)
        records = [rec for rec in history if rec.get("readiness") is not None]
        if window is not None:
            records = records[-window:]
        labels = [rec.get("ts") for rec in records]
        readiness = [float(rec.get("readiness", 0.0)) for rec in records]
        return {"labels": labels, "readiness": readiness}

    # ------------------------------------------------------------------
    def readiness_distribution_panel(
        self,
        telemetry: TelemetryBackend,
        workflow_id: str | None = None,
        *,
        scope: Scope | str = "local",
    ) -> Dict[str, List[float]]:
        """Return distributions for readiness and prediction errors."""

        history = telemetry.fetch_history(workflow_id, scope=scope)
        readiness = [
            float(rec["readiness"])
            for rec in history
            if rec.get("readiness") is not None
        ]
        errors = [
            abs(float(rec["predicted"]) - float(rec["actual"]))
            for rec in history
            if rec.get("predicted") is not None and rec.get("actual") is not None
        ]
        return {"readiness": readiness, "prediction_errors": errors}

    # ------------------------------------------------------------------
    def drift_instability_panel(
        self, tracker: _ROITrackerType | None = None, window: int | None = None
    ) -> Dict[str, Any]:
        """Return drift flags alongside instability metrics."""

        tracker = _ensure_tracker(tracker)
        drift_flags = tracker.drift_flags[-window:] if window else list(tracker.drift_flags)
        instability = tracker.metrics_history.get("instability", [])
        if window is not None:
            instability = instability[-window:]
        if len(instability) > len(drift_flags):
            instability = instability[-len(drift_flags):]
        elif len(drift_flags) > len(instability):
            drift_flags = drift_flags[-len(instability):]
        labels = list(range(len(drift_flags)))
        return {
            "labels": labels,
            "drift_flags": drift_flags,
            "instability": instability,
        }

    # ------------------------------------------------------------------
    def governance_panel(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent governance check outcomes."""

        if not GOVERNANCE_LOG.exists():
            return []
        lines = GOVERNANCE_LOG.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        return records[-limit:]

    # ------------------------------------------------------------------
    def relevancy_radar_panel(self, threshold: int = 5) -> List[Dict[str, Any]]:
        """Return modules with low relevancy scores and annotations."""

        module_dir = Path(__file__).resolve().parent
        local_metrics = module_dir / "sandbox_data" / "relevancy_metrics.json"
        if local_metrics.exists():
            metrics_path = local_metrics
        else:
            metrics_path = Path(resolve_path("sandbox_data/relevancy_metrics.json"))
            if not metrics_path.exists():
                return []
        try:
            with metrics_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            return []

        flags = flagged_modules()

        results: List[Dict[str, Any]] = []
        modules = set(data.keys()) | set(flags.keys())
        for mod in modules:
            info = data.get(mod, {})
            if not isinstance(info, dict):
                continue
            imports = int(info.get("imports", 0))
            executions = int(info.get("executions", 0))
            score = imports + executions
            annotation = str(info.get("annotation", ""))
            impact = float(info.get("impact", 0.0))
            flag = flags.get(mod)
            if score < threshold or annotation or flag:
                rec: Dict[str, Any] = {
                    "module": mod,
                    "imports": imports,
                    "executions": executions,
                    "score": score,
                    "impact": impact,
                    "flag": flag,
                }
                if annotation:
                    rec["annotation"] = annotation
                results.append(rec)
        results.sort(key=lambda r: r["score"])
        return results

    # ------------------------------------------------------------------
    def lineage_tree(self, workflow_id: int) -> List[Dict[str, Any]]:
        """Return the mutation lineage for ``workflow_id``.

        The tree is cached and rebuilt on demand.  :class:`MutationLineage` is
        used when available and falls back to :func:`mutation_logger.build_lineage`.
        """

        with self._lineage_lock:
            cached = self._lineage_trees.get(workflow_id)
        if cached is not None:
            return cached
        tree = _build_tree(workflow_id)
        with self._lineage_lock:
            self._lineage_trees[workflow_id] = tree
        return tree

    # ------------------------------------------------------------------
    def update_lineage_tree(self, workflow_id: int, tree: List[Dict[str, Any]]) -> None:
        """Update cached lineage tree and push to the update queue."""

        with self._lineage_lock:
            self._lineage_trees[workflow_id] = tree
        self._lineage_updates.put(tree)

    # ------------------------------------------------------------------
    def save_lineage_json(self, path: str | Path, workflow_id: int) -> None:
        """Persist the mutation lineage tree as JSON."""

        Path(path).write_text(
            json.dumps(self.lineage_tree(workflow_id), indent=2)
        )


def _ensure_tracker(tracker: _ROITrackerType | None) -> _ROITrackerType:
    if tracker is not None:
        return tracker
    return _instantiate_dependency("ROITracker", _get_roi_tracker_cls, _FallbackROITracker)


def refresh_dashboard(
    output: str | Path = "evaluation_dashboard.json",
    *,
    history: str = "roi_history.json",
    telemetry_db: str = "telemetry.db",
    scope: Scope | str = "local",
) -> Path:
    """Rebuild dashboard data and persist to ``output``.

    This helper loads the ROI prediction history and telemetry metrics to
    regenerate dashboard artefacts after each test cycle or deployment batch.
    """

    tracker = _instantiate_dependency("ROITracker", _get_roi_tracker_cls, _FallbackROITracker)
    try:
        tracker.load_history(history)
    except Exception:
        pass
    telemetry = _instantiate_dependency(
        "TelemetryBackend", _get_telemetry_backend_cls, _FallbackTelemetryBackend, telemetry_db
    )
    dash = EvaluationDashboard(
        _instantiate_dependency(
            "EvaluationManager", _get_evaluation_manager_cls, _FallbackEvaluationManager
        )
    )
    data = {
        "readiness_over_time": dash.readiness_chart(telemetry, scope=scope),
        "readiness_distribution": dash.readiness_distribution_panel(telemetry, scope=scope),
        "drift_instability": dash.drift_instability_panel(tracker),
    }
    out = Path(output)
    out.write_text(json.dumps(data, indent=2))
    return out


def main() -> None:  # pragma: no cover - CLI hook
    """CLI entry point used to refresh dashboard artefacts."""

    parser = argparse.ArgumentParser(description="Rebuild evaluation dashboard")
    parser.add_argument("--output", default="evaluation_dashboard.json")
    parser.add_argument("--history", default="roi_history.json")
    parser.add_argument("--telemetry", default="telemetry.db")
    parser.add_argument("--scope", default="local")
    args = parser.parse_args()
    path = refresh_dashboard(
        args.output,
        history=args.history,
        telemetry_db=args.telemetry,
        scope=args.scope,
    )
    print(f"dashboard refreshed: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()


__all__ = ["EvaluationDashboard", "refresh_dashboard", "main"]
