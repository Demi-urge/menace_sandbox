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
* resource costs and CPU/GPU usage gathered from :mod:`roi_tracker`
* counts of errors and repairs recorded by :mod:`error_bot`
* the GPT evaluation score, feedback and long-term outcomes for the cycle
* long-term ROI deltas derived from prediction events

The target vector contains the realised ROI outcome for the cycle calculated
from the ROI history.  Both functions remain available so existing code and
tests continue to operate.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import json
import sqlite3
import numpy as np

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope

if __package__:
    from .evolution_history_db import EvolutionHistoryDB
    from .evaluation_history_db import EvaluationHistoryDB
    from .roi_tracker import ROITracker
else:  # pragma: no cover - fallback for flat module layout
    from evolution_history_db import EvolutionHistoryDB  # type: ignore
    from evaluation_history_db import EvaluationHistoryDB  # type: ignore
    from roi_tracker import ROITracker  # type: ignore

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

__all__ = ["load_adaptive_roi_dataset", "build_dataset"]


# ---------------------------------------------------------------------------
def _parse_ts(ts: str) -> datetime:
    """Parse ISO formatted timestamps."""

    return datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
def _get_router(router: DBRouter | None = None) -> DBRouter:
    """Return an initialised :class:`DBRouter` instance."""

    return router or GLOBAL_ROUTER or init_db_router("default")


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
def _collect_eval_history(
    db: EvaluationHistoryDB,
) -> dict[str, list[tuple[datetime, float, float, float, float, float, str]]]:
    """Return evaluation scores and feedback grouped by engine.

    The returned mapping contains entries of the form
    ``(timestamp, cv_score, gpt_feedback_score, gpt_feedback_tokens,
    long_term_perf_delta, long_term_eval_outcome, gpt_feedback_text)``.
    ``long_term_eval_outcome`` falls back to ``0.0`` when the corresponding
    column is absent from :mod:`evaluation_history_db`. When available,
    ``gpt_feedback_embedding`` values are parsed from JSON and returned as a
    list of floats.
    """

    history: dict[str, list[tuple[datetime, float, float, float, float, float, str, list[float]]]] = {}
    cols = {
        row[1]
        for row in db.conn.execute("PRAGMA table_info(evaluation_history)").fetchall()
    }
    has_fb_score = "gpt_feedback_score" in cols
    has_fb_tokens = "gpt_feedback_tokens" in cols
    has_long = "long_term_delta" in cols
    has_long_outcome = "long_term_outcome" in cols
    has_fb_text = "gpt_feedback" in cols
    has_fb_emb = "gpt_feedback_embedding" in cols
    select_cols = ["cv_score", "ts"]
    if has_fb_score:
        select_cols.append("gpt_feedback_score")
    if has_fb_tokens:
        select_cols.append("gpt_feedback_tokens")
    if has_long:
        select_cols.append("long_term_delta")
    if has_long_outcome:
        select_cols.append("long_term_outcome")
    if has_fb_text:
        select_cols.append("gpt_feedback")
    if has_fb_emb:
        select_cols.append("gpt_feedback_embedding")
    query = (
        "SELECT " + ",".join(select_cols) + " FROM evaluation_history WHERE engine=? ORDER BY ts"
    )
    for eng in db.engines():
        cur = db.conn.execute(query, (eng,))
        records: list[tuple[datetime, float, float, float, float, float, str, list[float]]] = []
        for row in cur.fetchall():
            score = float(row[0])
            ts = row[1]
            idx = 2
            fb_score = float(row[idx]) if has_fb_score else 0.0
            idx += 1 if has_fb_score else 0
            tokens = float(row[idx]) if has_fb_tokens else 0.0
            idx += 1 if has_fb_tokens else 0
            long_term = float(row[idx]) if has_long else 0.0
            idx += 1 if has_long else 0
            long_outcome = float(row[idx]) if has_long_outcome else 0.0
            idx += 1 if has_long_outcome else 0
            fb_text = row[idx] if has_fb_text else ""
            idx += 1 if has_fb_text else 0
            if has_fb_emb:
                try:
                    emb_val = row[idx]
                    fb_emb = (
                        [float(v) for v in json.loads(emb_val)]
                        if emb_val not in (None, "")
                        else []
                    )
                except Exception:
                    fb_emb = []
            else:
                fb_emb = []
            records.append(
                (
                    _parse_ts(ts),
                    score,
                    fb_score,
                    tokens,
                    long_term,
                    long_outcome,
                    str(fb_text),
                    fb_emb,
                )
            )
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
def _collect_resource_metrics(tracker: ROITracker) -> dict[str, list[float]]:
    """Return tracked resource cost, CPU and GPU usage.

    ``ROITracker.resource_metrics`` may store tuples of either
    ``(cost, cpu, gpu)`` or ``(cpu, memory, disk, time, gpu)``.  This helper
    normalises the structure and returns a mapping with keys
    ``"resource_cost"``, ``"resource_cpu"`` and ``"resource_gpu"``.  Missing
    entries are padded with zeros.
    """

    metrics = {"resource_cost": [], "resource_cpu": [], "resource_gpu": []}
    for row in getattr(tracker, "resource_metrics", []) or []:
        try:
            if len(row) >= 5:
                cpu, _mem, _disk, cost, gpu = row[:5]
            elif len(row) == 3:
                cost, cpu, gpu = row
            else:
                cost, cpu, gpu = 0.0, 0.0, 0.0
            metrics["resource_cost"].append(float(cost))
            metrics["resource_cpu"].append(float(cpu))
            metrics["resource_gpu"].append(float(gpu))
        except Exception:
            metrics["resource_cost"].append(0.0)
            metrics["resource_cpu"].append(0.0)
            metrics["resource_gpu"].append(0.0)
    return metrics


# ---------------------------------------------------------------------------
def _collect_error_history(
    path: str | Path,
    router: DBRouter | None = None,
    *,
    scope: Scope | str = "local",
    source_menace_id: str | None = None,
) -> list[tuple[datetime, int, int]]:
    """Return cumulative error and repair counts ordered by timestamp."""

    records: list[tuple[datetime, int, int]] = []
    try:
        rtr = _get_router(router)
        menace_id = source_menace_id or rtr.menace_id
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            "SELECT ts, resolution_status FROM telemetry",
            clause,
        ) + " ORDER BY ts"
        conn = rtr.get_connection("telemetry")
        cur = conn.execute(query, params)
        err_cnt = 0
        rep_cnt = 0
        for ts, status in cur.fetchall():
            err_cnt += 1
            if status and str(status).lower() not in ("", "unresolved", "0"):
                rep_cnt += 1
            records.append((_parse_ts(ts), err_cnt, rep_cnt))
    except Exception:
        return []
    return records


# ---------------------------------------------------------------------------
def _collect_roi_event_extras(
    path: str | Path, router: DBRouter | None = None
) -> dict[str, list[float]]:
    """Return additional per-cycle metrics stored in ``roi_events.db``.

    Any columns on the ``roi_events`` table beyond ``action``, ``roi_before``,
    ``roi_after`` and ``ts`` are treated as numeric features and returned as a
    mapping of column name to a list of values ordered by timestamp. Missing or
    non-numeric entries are coerced to ``0.0``.  When the table or connection is
    unavailable an empty mapping is returned.
    """

    extras: dict[str, list[float]] = {}
    try:
        conn = _get_router(router).get_connection("roi_events")
        cols_cur = conn.execute("PRAGMA table_info(roi_events)")
        cols = [row[1] for row in cols_cur.fetchall()]
        base = {"action", "roi_before", "roi_after", "ts"}
        extra_cols = [c for c in cols if c not in base]
        if extra_cols:
            query = "SELECT " + ",".join(extra_cols) + " FROM roi_events ORDER BY ts"
            cur = conn.execute(query)
            rows = cur.fetchall()
            for idx, name in enumerate(extra_cols):
                extras[name] = []
                for r in rows:
                    try:
                        val = float(r[idx]) if r[idx] is not None else 0.0
                    except Exception:
                        val = 0.0
                    extras[name].append(val)
    except Exception:
        return {}
    return extras


# ---------------------------------------------------------------------------
def _ema(values: Sequence[float], span: int = 3) -> float:
    """Return exponential moving average of ``values``."""

    arr = list(values)
    if not arr:
        return 0.0
    alpha = 2.0 / float(span + 1)
    ema = float(arr[0])
    for v in arr[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return float(ema)


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
    *,
    router: DBRouter | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalise ROI training data including growth labels.

    Parameters
    ----------
    evolution_path:
        Path to the evolution history database.
    router:
        Optional :class:`DBRouter` for accessing evaluation history.

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
    eval_db = EvaluationHistoryDB(router=_get_router(router))

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
    roi_events_path: str | Path = "roi_events.db",
    errors_path: str | Path = "errors.db",
    horizons: Sequence[int] = (1, 3, 5),
    *,
    router: DBRouter | None = None,
    ema_span: int | None = 3,
    selected_features: Sequence[str] | None = None,
    return_feature_names: bool = False,
    export_path: str | Path | None = resolve_path("sandbox_data") / "adaptive_roi.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble a training dataset combining evolution, ROI and evaluation data.

    Parameters
    ----------
    evolution_path:
        Path to the evolution history SQLite database.
    roi_path:
        Path to the ROI history database (``action_roi`` table).
    roi_events_path:
        Path to the ROI prediction events database used to augment the
        training features.
    errors_path:
        Path to the ``error_bot`` telemetry database for error/repair counts.
    horizons:
        Sequence of future ROI steps to include as target columns.  The
        default ``(1, 3, 5)`` yields targets ``roi_{t+1}``, ``roi_{t+3}`` and
        ``roi_{t+5}``.  All listed horizons must be available for a row to be
        included.
    router:
        Optional :class:`DBRouter` instance for database access.
    ema_span:
        Optional span for an exponential moving average of ROI values up to the
        current cycle.  When provided an additional target column ``roi_ema`` is
        appended representing the smoothed ROI sequence.  Use ``None`` to skip
        this column.
    selected_features:
        Optional sequence of feature names to retain.  When ``None`` and a
        predictor metadata file (``sandbox_data/adaptive_roi.meta.json``)
        containing ``selected_features`` is present, those columns are selected
        automatically.
    export_path:
        Optional path for exporting the assembled feature matrix and targets as
        CSV.  When ``None`` the dataset is not written to disk.

    Returns
    -------
    tuple
        ``(features, targets, growth_types)`` where ``features`` is a matrix with columns
        ``[before_metric, after_metric, api_cost_delta, cpu_seconds_delta,
        success_rate_delta, gpt_score, gpt_feedback_score, gpt_feedback_tokens,
        long_term_perf_delta, long_term_eval_outcome, gpt_feedback_emb_*,
        resource_cost, resource_cpu_usage, resource_gpu_usage, error_count,
        repair_count, recovery_time, latency_error_rate,
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
        long_term_roi_reliability, cpu, memory, disk, time, gpu,
        predicted_roi_event, actual_roi_event, predicted_class_event,
        actual_class_event, prediction_confidence, predicted_horizon_delta,
        actual_horizon_delta]`` and ``targets`` contains multiple ROI columns as
        specified by ``horizons`` followed by an optional exponential moving
        average column ``roi_ema``.  ``growth_types`` labels the ROI curve of
        each cycle as ``"exponential"``, ``"linear"`` or ``"marginal"``.  When
        ``return_feature_names`` is true a fourth element containing the
    feature names is included.
    """

    if selected_features is None:
        try:
            meta_path = resolve_path("sandbox_data") / "adaptive_roi.meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                sel = meta.get("selected_features")
                if isinstance(sel, list) and sel:
                    selected_features = [str(s) for s in sel]
        except Exception:
            selected_features = None

    evo_db = EvolutionHistoryDB(evolution_path)
    router = _get_router(router)
    eval_db = EvaluationHistoryDB(router=router)
    roi_conn = router.get_connection("action_roi")

    roi_hist = _collect_roi_history(roi_conn)
    eval_hist = _collect_eval_history(eval_db)
    any_fb_text = any(
        rec[6]
        for records in eval_hist.values()
        for rec in records
        if len(rec) >= 7
    )
    any_fb_emb = any(
        rec[7]
        for records in eval_hist.values()
        for rec in records
        if len(rec) >= 8 and rec[7]
    )
    tracker_template = ROITracker()
    metric_names = sorted(
        set(tracker_template.metrics_history) | set(tracker_template.synergy_metrics_history)
    )
    resource_cols = ["cpu", "memory", "disk", "time", "gpu"]
    metric_names.extend(resource_cols)
    metrics_hist = _collect_metrics_history(roi_conn, metric_names)
    resource_hist = _collect_resource_metrics(tracker_template)
    error_hist = _collect_error_history(errors_path, router)
    event_extras = _collect_roi_event_extras(roi_events_path, router)
    event_errs = event_extras.pop("error_count", [])
    event_reps = event_extras.pop("repair_count", [])
    event_api_deltas = event_extras.pop("api_cost_delta", [])
    event_cpu_deltas = event_extras.pop("cpu_seconds_delta", [])
    event_sr_deltas = event_extras.pop("success_rate_delta", [])
    extra_feature_names = sorted(event_extras.keys())
    res_costs = resource_hist.get("resource_cost", [])
    res_cpus = resource_hist.get("resource_cpu", [])
    res_gpus = resource_hist.get("resource_gpu", [])
    err_idx = 0
    err_cnt = 0.0
    rep_cnt = 0.0
    horizons = sorted({int(h) for h in horizons if int(h) > 0}) or [1]
    max_h = max(horizons)
    embed_cache: dict[str, list[float]] = {}
    embedder = None
    embedding_dim = 0
    if any_fb_emb:
        try:
            sample_emb = next(
                rec[7]
                for records in eval_hist.values()
                for rec in records
                if len(rec) >= 8 and rec[7]
            )
            embedding_dim = len(sample_emb)
        except StopIteration:
            embedding_dim = 0
    elif any_fb_text:
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore

            _embedder = OpenAIEmbeddings()
            try:
                _dim_probe = _embedder.embed_query("probe")
                embedding_dim = len(_dim_probe)
                embedder = _embedder
            except Exception:
                embedding_dim = 0
        except Exception:  # pragma: no cover - graceful fallback
            embedding_dim = 0

    feature_names: list[str] = [
        "before_metric",
        "after_metric",
        "api_cost_delta",
        "cpu_seconds_delta",
        "success_rate_delta",
        "gpt_score",
        "gpt_feedback_score",
        "gpt_feedback_tokens",
        "long_term_perf_delta",
        "long_term_eval_outcome",
    ]
    if embedding_dim:
        feature_names.extend([f"gpt_feedback_emb_{i}" for i in range(embedding_dim)])
    feature_names.extend(
        [
            "resource_cost",
            "resource_cpu_usage",
            "resource_gpu_usage",
            "error_count",
            "repair_count",
        ]
    )
    feature_names.extend(extra_feature_names)
    feature_names.extend(metric_names)
    feature_names.extend(
        [
            "predicted_roi_event",
            "actual_roi_event",
            "predicted_class_event",
            "actual_class_event",
            "prediction_confidence",
            "predicted_horizon_delta",
            "actual_horizon_delta",
        ]
    )

    # load persisted prediction events --------------------------------------
    try:
        conn = router.get_connection("roi_prediction_events")
        try:
            cur = conn.execute(
                "SELECT predicted_roi, actual_roi, predicted_class, actual_class, "
                "confidence, predicted_horizons, actual_horizons FROM roi_prediction_events ORDER BY ts"
            )
            pred_rows = cur.fetchall()
        except sqlite3.OperationalError:
            cur = conn.execute(
                "SELECT predicted_roi, actual_roi, predicted_class, actual_class FROM roi_prediction_events ORDER BY ts"
            )
            pred_rows = [row + (None, None, None) for row in cur.fetchall()]
    except Exception:
        pred_rows = []

    pred_vals = [float(r[0]) for r in pred_rows]
    act_vals = [float(r[1]) for r in pred_rows]
    class_set = {
        str(r[2]) for r in pred_rows if r[2] is not None
    } | {str(r[3]) for r in pred_rows if r[3] is not None}
    class_map = {c: i for i, c in enumerate(sorted(class_set))}
    pred_codes = [class_map.get(str(r[2]), 0) for r in pred_rows]
    act_codes = [class_map.get(str(r[3]), 0) for r in pred_rows]
    conf_vals = [0.0 if r[4] is None else float(r[4]) for r in pred_rows]
    pred_h_deltas: list[float] = []
    act_h_deltas: list[float] = []
    for r in pred_rows:
        try:
            pred_h = json.loads(r[5]) if r[5] else []
        except Exception:
            pred_h = []
        try:
            act_h = json.loads(r[6]) if r[6] else []
        except Exception:
            act_h = []
        pred_h_deltas.append(
            float(pred_h[-1]) - float(pred_h[0]) if len(pred_h) >= 2 else 0.0
        )
        act_h_deltas.append(
            float(act_h[-1]) - float(act_h[0]) if len(act_h) >= 2 else 0.0
        )

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
        fb_score = 0.0
        fb_tokens = 0.0
        long_term = 0.0
        long_outcome = 0.0
        feedback_text = ""
        feedback_emb: list[float] = []
        for e_ts, score, fb, tokens, lt, lo, fb_txt, fb_emb in eval_list:
            if e_ts >= ts_dt:
                eval_score = score
                fb_score = fb
                fb_tokens = tokens
                long_term = lt
                long_outcome = lo
                feedback_text = fb_txt
                feedback_emb = fb_emb
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
        if (
            prev_roi is None
            or next_roi is None
            or next_idx is None
            or next_idx + max_h - 1 >= len(roi_list)
        ):
            continue

        if prev_idx is not None:
            cycle_idx = prev_idx
        api_delta = (
            event_api_deltas[cycle_idx]
            if cycle_idx < len(event_api_deltas)
            else next_roi[2] - prev_roi[2]
        )
        cpu_delta = (
            event_cpu_deltas[cycle_idx]
            if cycle_idx < len(event_cpu_deltas)
            else next_roi[3] - prev_roi[3]
        )
        sr_delta = (
            event_sr_deltas[cycle_idx]
            if cycle_idx < len(event_sr_deltas)
            else next_roi[4] - prev_roi[4]
        )
        try:
            roi_outcomes = [
                roi_list[next_idx + h - 1][1] - roi_list[next_idx + h - 1][2]
                for h in horizons
            ]
        except IndexError:
            continue
        if ema_span is not None:
            past_vals = [r[1] - r[2] for r in roi_list[: next_idx + 1]]
            roi_outcomes.append(_ema(past_vals, span=ema_span))

        row = [
            float(before),
            float(after),
            float(api_delta),
            float(cpu_delta),
            float(sr_delta),
            float(eval_score),
            float(fb_score),
            float(fb_tokens),
            float(long_term),
            float(long_outcome),
        ]
        if embedding_dim:
            emb_vec = [0.0] * embedding_dim
            if feedback_emb:
                emb_vec = feedback_emb[:embedding_dim]
            elif embedder and feedback_text:
                cached = embed_cache.get(feedback_text)
                if cached is None:
                    try:
                        cached = embedder.embed_query(feedback_text)
                    except Exception:  # pragma: no cover - fallback on error
                        cached = [0.0] * embedding_dim
                    embed_cache[feedback_text] = cached
                emb_vec = cached[:embedding_dim]
            row.extend(float(v) for v in emb_vec)
        res_cost = res_costs[cycle_idx] if cycle_idx < len(res_costs) else 0.0
        res_cpu = res_cpus[cycle_idx] if cycle_idx < len(res_cpus) else 0.0
        res_gpu = res_gpus[cycle_idx] if cycle_idx < len(res_gpus) else 0.0
        if event_errs:
            err_val = event_errs[cycle_idx] if cycle_idx < len(event_errs) else 0.0
            rep_val = event_reps[cycle_idx] if cycle_idx < len(event_reps) else 0.0
        else:
            while error_hist and err_idx < len(error_hist) and error_hist[err_idx][0] <= ts_dt:
                err_cnt = float(error_hist[err_idx][1])
                rep_cnt = float(error_hist[err_idx][2])
                err_idx += 1
            err_val = err_cnt
            rep_val = rep_cnt
        row.extend([float(res_cost), float(res_cpu), float(res_gpu), err_val, rep_val])
        for name in extra_feature_names:
            seq = event_extras.get(name, [])
            val = seq[cycle_idx] if cycle_idx < len(seq) else 0.0
            row.append(float(val))
        for name in metric_names:
            seq = metrics_hist.get(name, [])
            val = seq[cycle_idx] if cycle_idx < len(seq) else 0.0
            row.append(float(val))
        pred_val = pred_vals[cycle_idx] if cycle_idx < len(pred_vals) else 0.0
        act_val = act_vals[cycle_idx] if cycle_idx < len(act_vals) else 0.0
        pred_code = pred_codes[cycle_idx] if cycle_idx < len(pred_codes) else 0
        act_code = act_codes[cycle_idx] if cycle_idx < len(act_codes) else 0
        conf_val = conf_vals[cycle_idx] if cycle_idx < len(conf_vals) else 0.0
        pred_h_delta = (
            pred_h_deltas[cycle_idx] if cycle_idx < len(pred_h_deltas) else 0.0
        )
        act_h_delta = (
            act_h_deltas[cycle_idx] if cycle_idx < len(act_h_deltas) else 0.0
        )
        row.extend(
            [
                float(pred_val),
                float(act_val),
                float(pred_code),
                float(act_code),
                float(conf_val),
                float(pred_h_delta),
                float(act_h_delta),
            ]
        )
        features.append(row)
        targets.append([float(v) for v in roi_outcomes])
        roi_values = [r[1] - r[2] for r in roi_list[: next_idx + max_h]]
        growth_types.append(_label_growth(roi_values))
        cycle_idx += 1

    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    g = np.asarray(growth_types, dtype=object)
    if selected_features:
        idx = [i for i, name in enumerate(feature_names) if name in selected_features]
        if idx:
            X = X[:, idx]
            feature_names = [feature_names[i] for i in idx]
        else:
            feature_names = []

    # normalise all feature columns
    if X.size:
        for i in range(X.shape[1]):
            col = X[:, i]
            mean = col.mean()
            std = col.std()
            if std == 0:
                std = 1.0
            X[:, i] = (col - mean) / std

    if export_path:
        try:
            import pandas as pd

            tgt_names = [f"roi_t+{h}" for h in horizons]
            if ema_span is not None:
                tgt_names.append("roi_ema")
            df = pd.DataFrame(X, columns=feature_names)
            y_cols = len(tgt_names)
            y_arr = y.reshape(-1, y_cols) if y_cols > 1 else y.reshape(-1, 1)
            for i, name in enumerate(tgt_names):
                df[name] = y_arr[:, i]
            df["growth_type"] = g
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(export_path, index=False)
        except Exception:
            pass

    if return_feature_names:
        return X, y, g, feature_names
    return X, y, g
