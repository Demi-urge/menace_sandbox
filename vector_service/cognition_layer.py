"""High level orchestration layer around vector retrieval.

The :class:`CognitionLayer` combines the retriever, context builder,
patch logger and metrics database into a single convenience facade.  It
exposes a small API used by services that want to perform a retrieval
request and later record the outcome of a patch based on that retrieval.

An optional :class:`roi_tracker.ROITracker` instance can be supplied to
record ROI deltas for the origin databases contributing to a patch::

    from roi_tracker import ROITracker
    from vector_service.cognition_layer import CognitionLayer

    tracker = ROITracker()
    layer = CognitionLayer(roi_tracker=tracker)
    ctx, sid = layer.query("What is ROI?")
    # ... apply patch ...
    layer.record_patch_outcome(sid, True, contribution=1.0)

This automatically forwards the patch outcome to ``tracker`` so ROI
histories stay up to date.

A custom ranking model may also be supplied to override the default
retrieval ranker used by the context builder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import time
import json
from pathlib import Path

from .retriever import Retriever
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
from .decorators import log_and_measure

try:  # pragma: no cover - optional dependency
    from roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover - ROI tracking optional
    ROITracker = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from menace.unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - event bus optional
    UnifiedEventBus = None  # type: ignore


class CognitionLayer:
    """Tie together retrieval, context building and patch logging.

    Parameters
    ----------
    roi_tracker:
        Optional :class:`roi_tracker.ROITracker` instance used to update
        ROI histories when recording patch outcomes.  If not provided and
        the tracker can be imported, a default instance is created.
    ranking_model:
        Optional ranking model object passed to :class:`ContextBuilder`
        when one is not supplied directly.
    """

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder | None = None,
        patch_logger: PatchLogger | None = None,
        vector_metrics: VectorMetricsDB | None = None,
        roi_tracker: ROITracker | None = None,
        ranking_model: Any | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.vector_metrics = vector_metrics or VectorMetricsDB()
        self.roi_tracker = roi_tracker or (ROITracker() if ROITracker is not None else None)
        db_weights = None
        if self.vector_metrics is not None:
            try:
                db_weights = self.vector_metrics.get_db_weights()
            except Exception:  # pragma: no cover - best effort
                db_weights = None
        self.context_builder = context_builder or ContextBuilder(
            retriever=self.retriever,
            ranking_model=ranking_model,
            roi_tracker=self.roi_tracker,
            db_weights=db_weights,
        )
        if context_builder is not None and db_weights:
            try:
                if hasattr(self.context_builder, "refresh_db_weights"):
                    self.context_builder.refresh_db_weights(db_weights)  # type: ignore[attr-defined]
                elif hasattr(self.context_builder, "db_weights"):
                    self.context_builder.db_weights.update(db_weights)  # type: ignore[attr-defined]
            except Exception:
                pass
        self.patch_logger = patch_logger or PatchLogger(
            vector_metrics=self.vector_metrics,
            roi_tracker=self.roi_tracker,
        )
        if getattr(self.patch_logger, "roi_tracker", None) is not self.roi_tracker:
            self.patch_logger.roi_tracker = self.roi_tracker
        # Keep track of vectors by session so outcomes can be recorded later
        self._session_vectors: Dict[str, List[Tuple[str, str, float]]] = {}
        self._retrieval_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    def reload_ranker_model(self, model_path: str | "Path") -> None:
        """Reload ranking model on retriever and context builder."""

        try:
            self.retriever.reload_ranker_model(model_path)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            from pathlib import Path

            try:  # pragma: no cover - package relative import
                from .. import retrieval_ranker as _rr  # type: ignore
            except Exception:  # pragma: no cover - fallback
                import retrieval_ranker as _rr  # type: ignore

            self.context_builder.ranking_model = _rr.load_model(Path(model_path))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def reload_reliability_scores(self) -> None:
        """Refresh retriever reliability statistics."""

        try:
            self.retriever.reload_reliability_scores()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    def update_ranker(
        self,
        vectors: List[Tuple[str, str, float]],
        success: bool,
        roi_deltas: Dict[str, float] | None = None,
        risk_scores: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Update ranking weights based on patch outcome.

        The method aggregates updates by origin database so that future
        retrievals can prioritise sources that historically produced
        successful patches.  ROI deltas from ``roi_tracker`` or the provided
        ``roi_deltas`` mapping directly influence the weight adjustments.
        When ``risk_scores`` are supplied, origins that emitted vectors with
        high alignment severity or semantic alerts receive a negative penalty
        proportional to the score so risky databases are down-weighted.

        Weights are persisted in :class:`VectorMetricsDB` so the behaviour
        survives restarts.  The new weights are returned as a mapping.
        """

        if self.vector_metrics is None:
            return {}

        origins = {origin or "" for origin, _vec_id, _score in vectors}

        per_db: Dict[str, float] = roi_deltas.copy() if roi_deltas else {}

        # Always consult ROITracker for per-origin deltas so adaptive weights
        # incorporate the latest ROI information.  Tracker-derived deltas only
        # supplement explicit ``roi_deltas`` to avoid double counting.
        if self.roi_tracker is not None:
            try:
                raw = getattr(self.roi_tracker, "origin_db_deltas", {})
                raw = raw() if callable(raw) else raw
                for origin in origins:
                    if origin in per_db:
                        continue
                    vals = raw.get(origin)
                    if vals:
                        per_db[origin] = float(vals[-1])
            except Exception:
                pass

        if not per_db:
            delta = 0.1 if success else -0.1
            for origin in origins:
                per_db[origin] = per_db.get(origin, 0.0) + delta

        if risk_scores:
            for origin, sev in risk_scores.items():
                key = origin or ""
                penalty = abs(sev)
                per_db[key] = per_db.get(key, 0.0) - penalty

        updates: Dict[str, float] = {}
        for origin, change in per_db.items():
            try:
                updates[origin] = self.vector_metrics.update_db_weight(origin, change)
            except Exception:
                pass

        if updates:
            try:
                all_weights = self.vector_metrics.get_db_weights()
            except Exception:
                all_weights = updates
            try:
                if hasattr(self.context_builder, "refresh_db_weights"):
                    self.context_builder.refresh_db_weights(all_weights)  # type: ignore[attr-defined]
                elif hasattr(self.context_builder, "db_weights"):
                    try:
                        self.context_builder.db_weights.clear()  # type: ignore[attr-defined]
                        self.context_builder.db_weights.update(all_weights)  # type: ignore[attr-defined]
                    except Exception:
                        self.context_builder.db_weights = dict(all_weights)  # type: ignore[attr-defined]
            except Exception:
                pass

            # Persist full weight mapping so external services can reload after
            # restarts.  We merge into ``retrieval_ranker.json`` which already
            # tracks the current ranking model path.
            try:
                cfg = Path("retrieval_ranker.json")
                data = {"weights": {}}
                if cfg.exists():
                    try:
                        loaded = json.loads(cfg.read_text())
                        if isinstance(loaded, dict):
                            data.update(loaded)
                    except Exception:
                        pass
                weights = data.get("weights") or {}
                if not isinstance(weights, dict):
                    weights = {}
                for db, wt in all_weights.items():
                    try:
                        weights[str(db)] = float(wt)
                    except Exception:
                        continue
                data["weights"] = weights
                cfg.write_text(json.dumps(data))
            except Exception:
                pass
        return updates

    # ------------------------------------------------------------------
    @log_and_measure
    def query(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        session_id: str = "",
    ) -> Tuple[str, str]:
        """Retrieve context for *prompt* and store vector metrics.

        Returns a tuple of ``(context, session_id)``.  The ``session_id`` can
        later be passed to :meth:`record_patch_outcome` to update metrics once
        the patch succeeds or fails.
        """

        stats: Dict[str, Any] = {}
        metadata: Dict[str, List[Dict[str, Any]]]
        try:
            result = self.context_builder.build_context(
                prompt,
                top_k=top_k,
                include_vectors=True,
                session_id=session_id,
                return_stats=True,
                return_metadata=True,
            )
        except TypeError:  # pragma: no cover - older builders
            result = self.context_builder.build_context(
                prompt,
                top_k=top_k,
                include_vectors=True,
                session_id=session_id,
            )

        if isinstance(result, tuple) and len(result) == 5:
            context, sid, vectors, metadata, stats = result
        elif isinstance(result, tuple) and len(result) == 4:
            context, sid, vectors, stats = result
            metadata = {}
        else:  # pragma: no cover - defensive fallback
            context, sid, vectors = result  # type: ignore[misc]
            metadata = {}

        self._session_vectors[sid] = vectors
        meta_map: Dict[str, Dict[str, Any]] = {}
        meta_by_vid: Dict[str, Dict[str, Any]] = {}
        for entries in metadata.values():
            for entry in entries:
                vid = str(entry.get("vector_id") or entry.get("id") or entry.get("record_id") or "")
                if vid:
                    meta_by_vid[vid] = entry

        tokens = int(stats.get("tokens", 0))
        wall_time_ms = float(stats.get("wall_time_ms", 0.0))
        prompt_tokens = int(stats.get("prompt_tokens", 0))
        for rank, (origin, vec_id, score) in enumerate(vectors, start=1):
            key = f"{origin}:{vec_id}" if origin else vec_id
            entry = meta_by_vid.get(vec_id, {})
            meta_map[key] = entry
            vec_meta = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            age = 0.0
            if isinstance(entry, dict):
                if "age" in entry:
                    try:
                        age = float(entry["age"])
                    except Exception:
                        age = 0.0
                elif "age" in vec_meta:
                    try:
                        age = float(vec_meta["age"])
                    except Exception:
                        age = 0.0
                else:
                    ts = (
                        entry.get("timestamp")
                        or entry.get("ts")
                        or entry.get("created_at")
                        or vec_meta.get("timestamp")
                        or vec_meta.get("ts")
                        or vec_meta.get("created_at")
                    )
                    if ts is not None:
                        try:
                            age = max(0.0, time.time() - float(ts))
                        except Exception:
                            age = 0.0
            if self.vector_metrics is not None:
                try:  # Best effort metrics logging
                    self.vector_metrics.log_retrieval(
                        origin,
                        tokens=tokens,
                        wall_time_ms=wall_time_ms,
                        hit=True,
                        rank=rank,
                        contribution=0.0,
                        prompt_tokens=prompt_tokens,
                        session_id=sid,
                        vector_id=vec_id,
                        similarity=score,
                        context_score=score,
                        age=age,
                    )
                except Exception:
                    pass
        self._retrieval_meta[sid] = meta_map
        return context, sid

    # ------------------------------------------------------------------
    @log_and_measure
    async def query_async(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        session_id: str = "",
    ) -> Tuple[str, str]:
        """Asynchronous wrapper for :meth:`query`."""

        stats: Dict[str, Any] = {}
        metadata: Dict[str, List[Dict[str, Any]]]
        try:
            result = await self.context_builder.build_async(
                prompt,
                top_k=top_k,
                include_vectors=True,
                session_id=session_id,
                return_stats=True,
                return_metadata=True,
            )
        except TypeError:  # pragma: no cover - older builders
            result = await self.context_builder.build_async(
                prompt,
                top_k=top_k,
                include_vectors=True,
                session_id=session_id,
            )

        if isinstance(result, tuple) and len(result) == 5:
            context, sid, vectors, metadata, stats = result
        elif isinstance(result, tuple) and len(result) == 4:
            context, sid, vectors, stats = result
            metadata = {}
        else:  # pragma: no cover - defensive fallback
            context, sid, vectors = result  # type: ignore[misc]
            metadata = {}

        self._session_vectors[sid] = vectors
        meta_map: Dict[str, Dict[str, Any]] = {}
        meta_by_vid: Dict[str, Dict[str, Any]] = {}
        for entries in metadata.values():
            for entry in entries:
                vid = str(entry.get("vector_id") or entry.get("id") or entry.get("record_id") or "")
                if vid:
                    meta_by_vid[vid] = entry

        tokens = int(stats.get("tokens", 0))
        wall_time_ms = float(stats.get("wall_time_ms", 0.0))
        prompt_tokens = int(stats.get("prompt_tokens", 0))
        for rank, (origin, vec_id, score) in enumerate(vectors, start=1):
            key = f"{origin}:{vec_id}" if origin else vec_id
            entry = meta_by_vid.get(vec_id, {})
            meta_map[key] = entry
            vec_meta = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            age = 0.0
            if isinstance(entry, dict):
                if "age" in entry:
                    try:
                        age = float(entry["age"])
                    except Exception:
                        age = 0.0
                elif "age" in vec_meta:
                    try:
                        age = float(vec_meta["age"])
                    except Exception:
                        age = 0.0
                else:
                    ts = (
                        entry.get("timestamp")
                        or entry.get("ts")
                        or entry.get("created_at")
                        or vec_meta.get("timestamp")
                        or vec_meta.get("ts")
                        or vec_meta.get("created_at")
                    )
                    if ts is not None:
                        try:
                            age = max(0.0, time.time() - float(ts))
                        except Exception:
                            age = 0.0
            if self.vector_metrics is not None:
                try:  # Best effort metrics logging
                    self.vector_metrics.log_retrieval(
                        origin,
                        tokens=tokens,
                        wall_time_ms=wall_time_ms,
                        hit=True,
                        rank=rank,
                        contribution=0.0,
                        prompt_tokens=prompt_tokens,
                        session_id=sid,
                        vector_id=vec_id,
                        similarity=score,
                        context_score=score,
                        age=age,
                    )
                except Exception:
                    pass
        self._retrieval_meta[sid] = meta_map
        return context, sid

    # ------------------------------------------------------------------
    def record_patch_outcome(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
    ) -> None:
        """Forward patch outcome to :class:`PatchLogger`.

        ``session_id`` must match the value returned from :meth:`query`.  The
        stored vector identifiers will be passed to
        :meth:`PatchLogger.track_contributors`.  When an
        :class:`roi_tracker.ROITracker` is configured, ROI metrics for each
        origin database are updated automatically based on ``contribution`` or
        similarity scores.
        """

        vectors = self._session_vectors.pop(session_id, [])
        meta = self._retrieval_meta.pop(session_id, None)
        if not vectors:
            return
        vec_ids = [(f"{o}:{vid}", score) for o, vid, score in vectors]
        self.patch_logger.track_contributors(
            vec_ids,
            success,
            patch_id=patch_id,
            session_id=session_id,
            contribution=contribution,
            retrieval_metadata=meta,
        )

        risk_scores: Dict[str, float] = {}
        if meta:
            for origin, vid, _ in vectors:
                key = f"{origin}:{vid}" if origin else vid
                m = meta.get(key, {})
                sev = m.get("alignment_severity")
                alerts = m.get("semantic_alerts")
                risk = 0.0
                if sev:
                    try:
                        risk = max(risk, float(sev))
                    except Exception:
                        risk = max(risk, 1.0)
                if alerts:
                    risk = max(risk, 1.0)
                if risk:
                    ok = origin or ""
                    risk_scores[ok] = max(risk_scores.get(ok, 0.0), risk)

        roi_contribs: Dict[str, float] = {}
        used_tracker_deltas = False
        if self.roi_tracker is not None:
            try:  # pragma: no cover - best effort
                cur = self.vector_metrics.conn.execute(
                    """
                    SELECT db, tokens, contribution, hit
                      FROM vector_metrics
                     WHERE session_id=? AND event_type='retrieval'
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
                roi_after = sum(float(contrib or 0.0) for _db, _tok, contrib, _hit in rows)
                retrieval_metrics = [
                    {
                        "origin_db": str(db),
                        "tokens": float(contrib or 0.0),
                        "hit": bool(hit),
                    }
                    for db, _tokens, contrib, hit in rows
                ]
                self.roi_tracker.update(
                    0.0,
                    roi_after,
                    retrieval_metrics=retrieval_metrics,
                )
                deltas = getattr(self.roi_tracker, "origin_db_deltas", {})
                for origin, _vid, _score in vectors:
                    key = origin or ""
                    vals = deltas.get(key)
                    if vals:
                        roi_contribs[key] = abs(vals[-1])
                        used_tracker_deltas = True
            except Exception:
                pass

        if not roi_contribs:
            base = 0.0 if contribution is None else contribution
            for origin, _vid, score in vectors:
                roi = base if contribution is not None else score
                key = origin or ""
                roi_contribs[key] = roi_contribs.get(key, 0.0) + abs(roi)

        if self.roi_tracker is not None and roi_contribs and not used_tracker_deltas:
            metrics = {
                origin: {
                    "roi": roi,
                    "win_rate": 1.0 if success else 0.0,
                    "regret_rate": 0.0 if success else 1.0,
                }
                for origin, roi in roi_contribs.items()
            }
            try:
                self.roi_tracker.update_db_metrics(metrics)
            except Exception:
                pass

        roi_deltas = {
            origin: (roi if success else -roi) for origin, roi in roi_contribs.items()
        }
        updates = self.update_ranker(
            vectors, success, roi_deltas=roi_deltas, risk_scores=risk_scores
        )

        if roi_contribs:
            bus = getattr(self.patch_logger, "event_bus", None)
            if bus is None and UnifiedEventBus is not None:
                try:
                    bus = UnifiedEventBus()
                except Exception:
                    bus = None
            if bus is not None:
                for origin, roi in roi_contribs.items():
                    payload = {"db": origin, "roi": roi, "win": success}
                    if origin in updates:
                        payload["weight"] = updates[origin]
                    try:
                        bus.publish("retrieval:feedback", payload)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    async def record_patch_outcome_async(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
    ) -> None:
        """Asynchronous wrapper for :meth:`record_patch_outcome`."""

        vectors = self._session_vectors.pop(session_id, [])
        meta = self._retrieval_meta.pop(session_id, None)
        if not vectors:
            return
        vec_ids = [(f"{o}:{vid}", score) for o, vid, score in vectors]
        await self.patch_logger.track_contributors_async(
            vec_ids,
            success,
            patch_id=patch_id,
            session_id=session_id,
            contribution=contribution,
            retrieval_metadata=meta,
        )
        risk_scores: Dict[str, float] = {}
        if meta:
            for origin, vid, _ in vectors:
                key = f"{origin}:{vid}" if origin else vid
                m = meta.get(key, {})
                sev = m.get("alignment_severity")
                alerts = m.get("semantic_alerts")
                risk = 0.0
                if sev:
                    try:
                        risk = max(risk, float(sev))
                    except Exception:
                        risk = max(risk, 1.0)
                if alerts:
                    risk = max(risk, 1.0)
                if risk:
                    ok = origin or ""
                    risk_scores[ok] = max(risk_scores.get(ok, 0.0), risk)

        roi_contribs: Dict[str, float] = {}
        used_tracker_deltas = False
        if self.roi_tracker is not None:
            try:  # pragma: no cover - best effort
                cur = self.vector_metrics.conn.execute(
                    """
                    SELECT db, tokens, contribution, hit
                      FROM vector_metrics
                     WHERE session_id=? AND event_type='retrieval'
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
                roi_after = sum(float(contrib or 0.0) for _db, _tok, contrib, _hit in rows)
                retrieval_metrics = [
                    {
                        "origin_db": str(db),
                        "tokens": float(contrib or 0.0),
                        "hit": bool(hit),
                    }
                    for db, _tokens, contrib, hit in rows
                ]
                self.roi_tracker.update(
                    0.0,
                    roi_after,
                    retrieval_metrics=retrieval_metrics,
                )
                deltas = getattr(self.roi_tracker, "origin_db_deltas", {})
                for origin, _vid, _score in vectors:
                    key = origin or ""
                    vals = deltas.get(key)
                    if vals:
                        roi_contribs[key] = abs(vals[-1])
                        used_tracker_deltas = True
            except Exception:
                pass

        if not roi_contribs:
            base = 0.0 if contribution is None else contribution
            for origin, _vid, score in vectors:
                roi = base if contribution is not None else score
                key = origin or ""
                roi_contribs[key] = roi_contribs.get(key, 0.0) + abs(roi)

        if self.roi_tracker is not None and roi_contribs and not used_tracker_deltas:
            metrics = {
                origin: {
                    "roi": roi,
                    "win_rate": 1.0 if success else 0.0,
                    "regret_rate": 0.0 if success else 1.0,
                }
                for origin, roi in roi_contribs.items()
            }
            try:
                self.roi_tracker.update_db_metrics(metrics)
            except Exception:
                pass

        roi_deltas = {
            origin: (roi if success else -roi) for origin, roi in roi_contribs.items()
        }
        updates = self.update_ranker(
            vectors, success, roi_deltas=roi_deltas, risk_scores=risk_scores
        )
        if roi_contribs:
            bus = getattr(self.patch_logger, "event_bus", None)
            if bus is None and UnifiedEventBus is not None:
                try:
                    bus = UnifiedEventBus()
                except Exception:
                    bus = None
            if bus is not None:
                for origin, roi in roi_contribs.items():
                    payload = {"db": origin, "roi": roi, "win": success}
                    if origin in updates:
                        payload["weight"] = updates[origin]
                    try:
                        bus.publish("retrieval:feedback", payload)
                    except Exception:
                        pass


__all__ = ["CognitionLayer"]
