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
from typing import Tuple

import sqlite3
import numpy as np

from .evolution_history_db import EvolutionHistoryDB
from .evaluation_history_db import EvaluationHistoryDB

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
def load_adaptive_roi_dataset(
    evolution_path: str | Path = "evolution_history.db",
    evaluation_path: str | Path = "evaluation_history.db",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalise ROI training data.

    Parameters
    ----------
    evolution_path:
        Path to the evolution history database.
    evaluation_path:
        Path to the evaluation history database.

    Returns
    -------
    tuple
        ``(features, labels, passed_flags)`` where ``features`` is a normalised
        matrix of shape ``(n_samples, 2)`` containing ROI and performance
        deltas, ``labels`` is the normalised vector of evaluation CV scores and
        ``passed_flags`` is a binary vector indicating evaluation success.
    """

    evo_db = EvolutionHistoryDB(evolution_path)
    eval_db = EvaluationHistoryDB(evaluation_path)

    events = _collect_events(evo_db)

    features: list[list[float]] = []
    labels: list[float] = []
    passed: list[int] = []

    for engine in eval_db.engines():
        history = eval_db.history(engine, limit=1000000)
        ev_list = events.get(engine)
        if not ev_list:
            continue
        for cv_score, ts, pflag, _ in history:
            ts_dt = _parse_ts(ts)
            latest: tuple[datetime, float, float] | None = None
            for ev in ev_list:
                if ev[0] <= ts_dt:
                    latest = ev
                else:
                    break
            if latest is None:
                continue
            features.append([latest[1], latest[2]])
            labels.append(float(cv_score))
            passed.append(int(pflag))

    if not features:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
        )

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    p = np.asarray(passed, dtype=int)

    # normalise features and labels
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std() or 1.0
    y = (y - y_mean) / y_std

    return X, y, p


# ---------------------------------------------------------------------------
def build_dataset(
    evolution_path: str | Path = "evolution_history.db",
    roi_path: str | Path = "roi.db",
    evaluation_path: str | Path = "evaluation_history.db",
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble a training dataset combining evolution, ROI and evaluation data.

    Parameters
    ----------
    evolution_path:
        Path to the evolution history SQLite database.
    roi_path:
        Path to the ROI history database (``action_roi`` table).
    evaluation_path:
        Path to the evaluation history database containing GPT scores.

    Returns
    -------
    tuple
        ``(features, targets)`` where ``features`` is a matrix with columns
        ``[before_metric, after_metric, api_cost_delta, cpu_seconds_delta,
        success_rate_delta, gpt_score]`` and ``targets`` is the vector of ROI
        outcomes ``(revenue - api_cost)`` for each cycle.
    """

    evo_db = EvolutionHistoryDB(evolution_path)
    eval_db = EvaluationHistoryDB(evaluation_path)
    roi_conn = sqlite3.connect(roi_path)

    roi_hist = _collect_roi_history(roi_conn)
    eval_hist = _collect_eval_history(eval_db)

    features: list[list[float]] = []
    targets: list[float] = []

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
        for rec in roi_list:
            if rec[0] <= ts_dt:
                prev_roi = rec
            elif rec[0] > ts_dt and next_roi is None:
                next_roi = rec
                break
        if prev_roi is None or next_roi is None:
            continue

        api_delta = next_roi[2] - prev_roi[2]
        cpu_delta = next_roi[3] - prev_roi[3]
        sr_delta = next_roi[4] - prev_roi[4]
        roi_outcome = next_roi[1] - next_roi[2]

        features.append(
            [
                float(before),
                float(after),
                float(api_delta),
                float(cpu_delta),
                float(sr_delta),
                float(eval_score),
            ]
        )
        targets.append(float(roi_outcome))

    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
