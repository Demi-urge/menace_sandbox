from __future__ import annotations

"""Utilities for assembling ROI training datasets.

Historically this module exposed :func:`load_adaptive_roi_dataset` which only
combined evolution and evaluation history.  The new
:func:`build_dataset` entry point augments this by also incorporating resource
usage deltas from ``roi.db`` (see :mod:`pre_execution_roi_bot`).  Each row in
the returned feature matrix corresponds to a single improvement cycle and
contains:

* pre and post performance metrics
* changes in resource usage (API cost, CPU seconds, success rate)
* the GPT evaluation score for the cycle

The target vector contains the realised ROI outcome for the cycle calculated
from the ROI history.  Both functions remain available so existing code and
tests continue to operate.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Sequence

import sqlite3
import numpy as np

from .evolution_history_db import EvolutionHistoryDB
from .evaluation_history_db import EvaluationHistoryDB
from .roi_tracker import ROITracker

__all__ = ["load_adaptive_roi_dataset", "build_dataset"]


# ---------------------------------------------------------------------------
def _parse_ts(ts: str) -> datetime:
    """Parse ISO formatted timestamps."""

    return datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
def _collect_events(db: EvolutionHistoryDB) -> dict[str, list[tuple[datetime, float, float]]]:
    """Collect evolution events grouped by action.

    Returns a mapping of action name to a list of tuples containing
    ``(timestamp, roi, performance_delta)`` sorted by timestamp.
    """

    events: dict[str, list[tuple[datetime, float, float]]] = defaultdict(list)
    cur = db.conn.execute(
        "SELECT action, before_metric, after_metric, roi, ts FROM evolution_history"
    )
    for action, before, after, roi, ts in cur.fetchall():
        before = float(before)
        after = float(after)
        perf_delta = after - before
        events[action].append((_parse_ts(ts), float(roi), perf_delta))
    for ev_list in events.values():
        ev_list.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
def _collect_roi_history(conn: sqlite3.Connection) -> dict[
    str, list[tuple[datetime, float, float, float, float]]
]:
    """Return ROI records grouped by action.

    Each list entry is ``(timestamp, revenue, api_cost, cpu_seconds, success_rate)``
    sorted by timestamp.
    """

    records: dict[str, list[tuple[datetime, float, float, float, float]]] = defaultdict(list)
    cur = conn.execute(
        "SELECT action, revenue, api_cost, cpu_seconds, success_rate, ts FROM action_roi"
    )
    for action, rev, api, cpu, sr, ts in cur.fetchall():
        records[action].append(
            (_parse_ts(ts), float(rev), float(api), float(cpu), float(sr))
        )
    for recs in records.values():
        recs.sort(key=lambda r: r[0])
    return records


# ---------------------------------------------------------------------------
def _collect_eval_history(db: EvaluationHistoryDB) -> dict[str, list[tuple[datetime, float]]]:
    """Return evaluation scores grouped by engine."""

    history: dict[str, list[tuple[datetime, float]]] = {}
    for eng in db.engines():
        records = [
            (_parse_ts(ts), float(score))
            for score, ts, _passed, _err in db.history(eng, limit=1000000)
        ]
        records.sort(key=lambda r: r[0])
        history[eng] = records
    return history


# ---------------------------------------------------------------------------
def _collect_metrics_history(
    conn: sqlite3.Connection, metric_names: list[str]
) -> dict[str, list[float]]:
    """Return ordered metric histories for known ``metric_names``."""

    metrics: dict[str, list[float]] = {name: [] for name in metric_names}
    try:
        cur = conn.execute(
            "SELECT metric, value FROM metrics_history ORDER BY rowid"
        )
    except Exception:
        return metrics
    for name, value in cur.fetchall():
        if name in metrics:
            metrics[name].append(0.0 if value is None else float(value))
    return metrics


# ---------------------------------------------------------------------------
def _label_growth(values: Sequence[float]) -> str:
    """Label ``values`` as ``exponential``, ``linear`` or ``marginal``.

    This heuristic is used to generate training labels for the supervised
    classifier.  It examines the first and second differences of the ROI
    sequence to determine the overall trend.  The predictor itself no longer
    relies on these thresholds for inference.
    """

    if len(values) < 2:
        return "marginal"
    arr = np.asarray(list(values), dtype=float)
    first_diff = np.diff(arr)
    slope = float(first_diff.mean()) if first_diff.size else 0.0
    curvature = float(np.diff(first_diff).mean()) if first_diff.size > 1 else 0.0
    slope_thr = float(np.mean(np.abs(first_diff)) * 0.5) if first_diff.size else 0.05
    curv_thr = (
        float(np.mean(np.abs(np.diff(first_diff))) * 0.5)
        if first_diff.size > 1
        else 0.01
    )
    if slope > slope_thr and curvature > curv_thr:
        return "exponential"
    if abs(slope) > slope_thr:
        return "linear"
    return "marginal"


# ---------------------------------------------------------------------------
def load_adaptive_roi_dataset(
    evolution_path: str | Path = "evolution_history.db",
    evaluation_path: str | Path = "evaluation_history.db",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalise ROI training data including growth labels.

    Parameters
    ----------
    evolution_path:
        Path to the evolution history database.
    evaluation_path:
        Path to the evaluation history database.

    Returns
    -------
    tuple
        ``(features, labels, passed_flags, growth_labels)`` where ``features`` is
        a normalised matrix of shape ``(n_samples, 2)`` containing ROI and
        performance deltas, ``labels`` is the normalised vector of evaluation CV
        scores, ``passed_flags`` is a binary vector indicating evaluation
        success and ``growth_labels`` provides the heuristic growth class for
        each sample.
    """

    evo_db = EvolutionHistoryDB(evolution_path)
    eval_db = EvaluationHistoryDB(evaluation_path)

    events = _collect_events(evo_db)

    features: list[list[float]] = []
    labels: list[float] = []
    passed: list[int] = []
    growth: list[str] = []

    for engine in eval_db.engines():
        history = eval_db.history(engine, limit=1000000)
        ev_list = events.get(engine)
        if not ev_list:
            continue
        for cv_score, ts, pflag, _ in history:
            ts_dt = _parse_ts(ts)
            latest: tuple[datetime, float, float] | None = None
            roi_seq: list[float] = []
            for ev in ev_list:
                if ev[0] <= ts_dt:
                    latest = ev
                    roi_seq.append(ev[1])
                else:
                    break
            if latest is None:
                continue
            features.append([latest[1], latest[2]])
            labels.append(float(cv_score))
            passed.append(int(pflag))
            growth.append(_label_growth(roi_seq))

    if not features:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=object),
        )

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    p = np.asarray(passed, dtype=int)
    g = np.asarray(growth, dtype=object)

    # normalise features and labels
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std() or 1.0
    y = (y - y_mean) / y_std

    return X, y, p, g


# ---------------------------------------------------------------------------
def build_dataset(
    evolution_path: str | Path = "evolution_history.db",
    roi_path: str | Path = "roi.db",
    evaluation_path: str | Path = "evaluation_history.db",
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble a training dataset combining evolution, ROI and evaluation data.

    Parameters
    ----------
    evolution_path:
        Path to the evolution history SQLite database.
    roi_path:
        Path to the ROI history database (``action_roi`` table).
    evaluation_path:
        Path to the evaluation history database containing GPT scores.
    horizon:
        Number of future ROI outcomes ``roi_{t+1} .. roi_{t+h}`` to include as
        target columns.

    Returns
    -------
    tuple
        ``(features, targets, growth_types)`` where ``features`` is a matrix with columns
        ``[before_metric, after_metric, api_cost_delta, cpu_seconds_delta,
        success_rate_delta, gpt_score, recovery_time, latency_error_rate,
        hostile_failures, misuse_failures, concurrency_throughput,
        synergy_adaptability, synergy_recovery_time, synergy_discrepancy_count,
        synergy_gpu_usage, synergy_cpu_usage, synergy_memory_usage,
        synergy_long_term_lucrativity, synergy_shannon_entropy,
        synergy_flexibility, synergy_energy_consumption, synergy_profitability,
        synergy_revenue, synergy_projected_lucrativity, synergy_maintainability,
        synergy_code_quality, synergy_network_latency, synergy_throughput,
        synergy_latency_error_rate, synergy_hostile_failures,
        synergy_misuse_failures, synergy_concurrency_throughput,
        synergy_risk_index, synergy_safety_rating, synergy_efficiency,
        synergy_antifragility, synergy_resilience, synergy_security_score,
        synergy_reliability, synergy_roi_reliability, roi_reliability,
        long_term_roi_reliability, cpu, memory, disk, time, gpu]`` and
        ``targets`` is the vector of ROI outcomes ``(revenue - api_cost)`` for
        each cycle and ``growth_types`` is a vector labelling the ROI curve of each
        cycle as ``"exponential"``, ``"linear"`` or ``"marginal"``.
    """

    evo_db = EvolutionHistoryDB(evolution_path)
    eval_db = EvaluationHistoryDB(evaluation_path)
    roi_conn = sqlite3.connect(roi_path)

    roi_hist = _collect_roi_history(roi_conn)
    eval_hist = _collect_eval_history(eval_db)
    tracker_template = ROITracker()
    metric_names = sorted(
        set(tracker_template.metrics_history) | set(tracker_template.synergy_metrics_history)
    )
    resource_cols = ["cpu", "memory", "disk", "time", "gpu"]
    metric_names.extend(resource_cols)
    metrics_hist = _collect_metrics_history(roi_conn, metric_names)

    features: list[list[float]] = []
    targets: list[list[float]] = []
    growth_types: list[str] = []
    cycle_idx = 0

    cur = evo_db.conn.execute(
        "SELECT action, before_metric, after_metric, ts FROM evolution_history ORDER BY ts"
    )
    for action, before, after, ts in cur.fetchall():
        ts_dt = _parse_ts(ts)

        # find evaluation score after the event
        eval_list = eval_hist.get(action)
        if not eval_list:
            continue
        eval_score = None
        for e_ts, score in eval_list:
            if e_ts >= ts_dt:
                eval_score = score
                break
        if eval_score is None:
            continue

        roi_list = roi_hist.get(action)
        if not roi_list:
            continue
        prev_roi = None
        next_roi = None
        prev_idx: int | None = None
        next_idx: int | None = None
        for idx, rec in enumerate(roi_list):
            if rec[0] <= ts_dt:
                prev_roi = rec
                prev_idx = idx
            elif rec[0] > ts_dt and next_roi is None:
                next_roi = rec
                next_idx = idx
                break
        if prev_roi is None or next_roi is None or next_idx is None:
            continue

        api_delta = next_roi[2] - prev_roi[2]
        cpu_delta = next_roi[3] - prev_roi[3]
        sr_delta = next_roi[4] - prev_roi[4]
        row = [
            float(before),
            float(after),
            float(api_delta),
            float(cpu_delta),
            float(sr_delta),
            float(eval_score),
        ]
        for name in metric_names:
            seq = metrics_hist.get(name, [])
            val = seq[cycle_idx] if cycle_idx < len(seq) else 0.0
            row.append(float(val))
        # collect ROI outcomes for the requested horizon
        seq: list[float] = []
        for i in range(horizon):
            idx = next_idx + i
            if idx >= len(roi_list):
                seq = []
                break
            roi_rec = roi_list[idx]
            seq.append(float(roi_rec[1] - roi_rec[2]))
        if not seq:
            continue
        features.append(row)
        targets.append(seq)
        roi_values = [r[1] - r[2] for r in roi_list[: next_idx + horizon]]
        growth_types.append(_label_growth(roi_values))
        cycle_idx += 1

    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    g = np.asarray(growth_types, dtype=object)

    # normalise all feature columns
    if X.size:
        for i in range(X.shape[1]):
            col = X[:, i]
            mean = col.mean()
            std = col.std()
            if std == 0:
                std = 1.0
            X[:, i] = (col - mean) / std

    return X, y, g
