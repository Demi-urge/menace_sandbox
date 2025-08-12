from __future__ import annotations

"""Utilities for visualising evaluation results."""

import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List
import types

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

from .evaluation_manager import EvaluationManager
from .roi_tracker import ROITracker


def _build_tree(workflow_id: int) -> List[Dict[str, Any]]:
    """Helper returning the lineage tree for ``workflow_id``."""

    try:
        from .mutation_lineage import MutationLineage

        return MutationLineage().build_tree(workflow_id)
    except Exception:  # pragma: no cover - best effort fallback
        from .mutation_logger import build_lineage

        return build_lineage(workflow_id)


class EvaluationDashboard:
    """Render :class:`EvaluationManager` history as data frames and weights."""

    def __init__(self, manager: EvaluationManager) -> None:
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
    def roi_prediction_panel(
        self, tracker: ROITracker, window: int | None = None
    ) -> Dict[str, Any]:
        """Return ROI prediction stats including trend and confusion metrics."""

        return tracker.prediction_summary(window)

    # ------------------------------------------------------------------
    def roi_prediction_chart(
        self, tracker: ROITracker, window: int | None = None
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

        total = len(tracker.predicted_roi)
        preds = tracker.predicted_roi[-window:] if window else tracker.predicted_roi
        acts = tracker.actual_roi[-len(preds) :]
        start = total - len(preds)
        labels = list(range(start, start + len(preds)))
        return {"labels": labels, "predicted": preds, "actual": acts}

    # ------------------------------------------------------------------
    def roi_prediction_events_panel(
        self, tracker: ROITracker, window: int | None = None
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

        mae_by_horizon = (
            tracker.horizon_mae_history[-1] if tracker.horizon_mae_history else {}
        )
        drift_flags = tracker.drift_flags[-window:] if window else tracker.drift_flags
        predictor = getattr(tracker, "_adaptive_predictor", None)
        drift_metrics = getattr(predictor, "drift_metrics", {}) if predictor else {}
        if not drift_metrics:
            drift_metrics = getattr(tracker, "drift_metrics", {})
        return {
            "mae_by_horizon": mae_by_horizon,
            "growth_class_accuracy": tracker.classification_accuracy(window),
            "drift_flags": drift_flags,
            "growth_type_accuracy": drift_metrics.get("accuracy", 0.0),
            "drift_metrics": drift_metrics,
        }

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


__all__ = ["EvaluationDashboard"]
