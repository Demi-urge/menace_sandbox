from __future__ import annotations

"""Utility helpers for ranking retrieved patches."""

from typing import Iterable, List, Tuple, Any, Optional
import time

try:  # pragma: no cover - optional dependency
    from code_database import PatchHistoryDB  # type: ignore
except Exception:  # pragma: no cover
    PatchHistoryDB = None  # type: ignore


def _roi_from_tracker(vector_id: str, tracker: Any) -> Optional[float]:
    """Attempt to pull ROI metric for *vector_id* from *tracker*."""
    if tracker is None:
        return None
    for attr in ("get_patch_roi", "patch_roi", "get_roi"):
        fn = getattr(tracker, attr, None)
        if callable(fn):
            try:
                return float(fn(vector_id))
            except Exception:
                return None
    return None


def _roi_recency_from_db(vector_id: str, db: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(roi_delta, recency_factor)`` for *vector_id* using *db*."""
    if db is None or not vector_id:
        return None, None
    try:
        rows = db.find_by_vector(vector_id)
        if rows:
            _pid, record = rows[0]
            roi = getattr(record, "roi_delta", None)
            ts = getattr(record, "ts", None)
            recency = None
            if ts is not None:
                age = max(0.0, time.time() - float(ts))
                recency = 1.0 / (1.0 + age / 86400.0)
            return roi, recency
    except Exception:
        return None, None
    return None, None


def rank_patches(
    entries: Iterable[Any],
    *,
    roi_tracker: Any | None = None,
    patch_db: Any | None = None,
    similarity_weight: float = 1.0,
    roi_weight: float = 1.0,
    recency_weight: float = 1.0,
    exclude_tags: Iterable[str] | None = None,
) -> Tuple[List[Any], float]:
    """Return ranked entries with a confidence score.

    Parameters
    ----------
    entries:
        Iterable of ``_ScoredEntry`` instances.
    roi_tracker:
        Optional ROI tracker used to look up ROI metrics when absent from
        entries.
    patch_db:
        Optional :class:`PatchHistoryDB` instance for ROI/recency lookup.
    similarity_weight, roi_weight, recency_weight:
        Multipliers applied to similarity, ROI and recency respectively.
    exclude_tags:
        Iterable of tags; any entry containing one of these is skipped.

    Returns
    -------
    ranked:
        List of entries sorted by weighted score.
    confidence:
        Highest weighted score observed, useful for fallback decisions.
    """

    db = patch_db
    if db is None and PatchHistoryDB is not None:  # pragma: no cover - best effort
        try:
            db = PatchHistoryDB()
        except Exception:
            db = None
    ranked: List[Any] = []
    best = 0.0
    excluded = set(exclude_tags or [])
    for item in entries:
        if excluded:
            meta = getattr(item, "metadata", None)
            if isinstance(meta, dict):
                tags = meta.get("tags") or []
                tag_set = (
                    {str(t) for t in tags}
                    if isinstance(tags, (list, tuple, set))
                    else {str(tags)} if tags else set()
                )
                if tag_set & excluded:
                    continue
        sim = float(getattr(item.entry, "get", lambda k, d=None: d)("similarity", getattr(item, "score", 0.0)))
        roi = item.entry.get("roi") if isinstance(item.entry, dict) else None
        recency = item.entry.get("recency") if isinstance(item.entry, dict) else None
        if roi is None:
            roi = _roi_from_tracker(item.vector_id, roi_tracker)
        if (roi is None or recency is None) and db is not None:
            db_roi, db_rec = _roi_recency_from_db(item.vector_id, db)
            if roi is None:
                roi = db_roi
            if recency is None:
                recency = db_rec
        score = sim * similarity_weight
        if roi is not None:
            score += float(roi) * roi_weight
            if isinstance(item.entry, dict):
                item.entry["roi"] = float(roi)
        if recency is not None:
            score += float(recency) * recency_weight
            if isinstance(item.entry, dict):
                item.entry["recency"] = float(recency)
        item.score = score
        ranked.append(item)
        if score > best:
            best = score
    ranked.sort(key=lambda e: e.score, reverse=True)
    return ranked, best


__all__ = ["rank_patches"]
