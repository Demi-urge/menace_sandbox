from __future__ import annotations

"""Utility for assembling training data from evolution and evaluation history.

This module extracts ROI and performance deltas from :mod:`evolution_history_db`
entries and pairs them with evaluation feedback from
:mod:`evaluation_history_db`.  The resulting data is normalised and returned as
NumPy arrays suitable for use in machine learning workflows.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from .evolution_history_db import EvolutionHistoryDB
from .evaluation_history_db import EvaluationHistoryDB

__all__ = ["load_adaptive_roi_dataset"]


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
