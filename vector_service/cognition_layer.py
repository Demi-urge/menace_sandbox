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

from typing import Any, Dict, List, Mapping, Tuple
import time
import asyncio
import json
import logging
import os
from pathlib import Path

from dynamic_path_router import resolve_path

from .retriever import Retriever, PatchRetriever
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
from .decorators import log_and_measure
from .embedding_backfill import schedule_backfill


logger = logging.getLogger(__name__)

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
        patch_retriever: PatchRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        patch_logger: PatchLogger | None = None,
        vector_metrics: VectorMetricsDB | None = None,
        event_bus: "UnifiedEventBus" | None = None,
        roi_tracker: ROITracker | None = None,
        ranking_model: Any | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.patch_retriever = patch_retriever or PatchRetriever()
        self.vector_metrics = vector_metrics or VectorMetricsDB()
        self.roi_tracker = roi_tracker or (ROITracker() if ROITracker is not None else None)
        self.event_bus = event_bus or getattr(patch_logger, "event_bus", None)
        db_weights = None
        if self.vector_metrics is not None:
            try:
                db_weights = self.vector_metrics.get_db_weights()
            except Exception:  # pragma: no cover - best effort
                db_weights = None
        self._db_weights: Dict[str, float] = dict(db_weights or {})
        if context_builder is not None:
            self.context_builder = context_builder
        else:
            self.context_builder = ContextBuilder(
                retriever=self.retriever,
                patch_retriever=self.patch_retriever,
                ranking_model=ranking_model,
                roi_tracker=self.roi_tracker,
                db_weights=db_weights,
            )
        if (
            context_builder is not None
            and getattr(context_builder, "patch_retriever", None) is None
        ):
            try:
                context_builder.patch_retriever = self.patch_retriever
            except Exception:
                pass
        if context_builder is not None and db_weights:
            try:
                if hasattr(self.context_builder, "refresh_db_weights"):
                    self.context_builder.refresh_db_weights(
                        db_weights
                    )  # type: ignore[attr-defined]
                elif hasattr(self.context_builder, "db_weights"):
                    self.context_builder.db_weights.update(
                        db_weights
                    )  # type: ignore[attr-defined]
            except Exception:
                logger.exception(
                    "Failed to refresh context builder db weights"
                )
        self.patch_logger = patch_logger or PatchLogger(
            vector_metrics=self.vector_metrics,
            roi_tracker=self.roi_tracker,
            event_bus=self.event_bus,
        )
        if getattr(self.patch_logger, "roi_tracker", None) is not self.roi_tracker:
            self.patch_logger.roi_tracker = self.roi_tracker
        if getattr(self.patch_logger, "event_bus", None) is not self.event_bus:
            try:
                self.patch_logger.event_bus = self.event_bus
            except Exception:
                pass
        # Keep track of vectors by session so outcomes can be recorded later
        self._session_vectors: Dict[str, List[Tuple[str, str, float]]] = {}
        self._retrieval_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if self.vector_metrics is not None:
            try:
                pending = self.vector_metrics.load_sessions()
                for sid, (vecs, meta) in pending.items():
                    self._session_vectors[sid] = vecs
                    self._retrieval_meta[sid] = meta
            except Exception:
                logger.exception("Failed to load pending sessions from metrics DB")

    # ------------------------------------------------------------------
    def reload_ranker_model(
        self,
        model_path: str | "Path" | None = None,
        *,
        roi_delta: float | None = None,
        risk_penalty: float | None = None,
    ) -> None:
        """Reload ranking model on retriever and context builder.

        When ``model_path`` is ``None`` the method attempts to read the path
        from ``retrieval_ranker.json`` so services can simply call this method
        after a scheduler-triggered retrain.  Passing ``roi_delta`` or
        ``risk_penalty`` allows the method to trigger an asynchronous retrain
        when thresholds configured via environment variables are exceeded.
        """

        if roi_delta is not None or risk_penalty is not None:
            try:
                roi_thresh = float(os.getenv("RANKER_SCHEDULER_ROI_THRESHOLD", "0") or 0.0)
            except Exception:
                roi_thresh = 0.0
            try:
                risk_thresh = float(os.getenv("RANKER_SCHEDULER_RISK_THRESHOLD", "0") or 0.0)
            except Exception:
                risk_thresh = 0.0
            if (
                roi_delta is not None and roi_thresh and abs(roi_delta) >= roi_thresh
            ) or (
                risk_penalty is not None and risk_thresh and abs(risk_penalty) >= risk_thresh
            ):
                try:
                    from analytics import ranker_scheduler as rs

                    rs.schedule_retrain([self])
                except Exception:
                    logger.exception("Failed to schedule ranker retrain")

        if not model_path:
            try:
                cfg = json.loads(resolve_path("retrieval_ranker.json").read_text())
                if isinstance(cfg, dict):
                    model_path = cfg.get("current")
            except Exception:
                model_path = None
        if not model_path:
            logger.warning("No ranking model path available to reload")
            return

        try:
            model_path = resolve_path(str(model_path))
        except Exception:
            logger.exception("Failed to resolve model path %s", model_path)
            return

        try:
            self.retriever.reload_ranker_model(model_path)  # type: ignore[arg-type]
        except Exception:
            logger.exception("Failed to reload ranker model on retriever")
        try:
            try:  # pragma: no cover - package relative import
                from .. import retrieval_ranker as _rr  # type: ignore
            except Exception:  # pragma: no cover - fallback
                import retrieval_ranker as _rr  # type: ignore

            self.context_builder.ranking_model = _rr.load_model(model_path)
        except Exception:
            logger.exception("Failed to load ranking model from %s", model_path)

        weights: Dict[str, float] = {}
        if self.vector_metrics is not None:
            try:
                weights = self.vector_metrics.get_db_weights()
            except Exception:
                weights = {}
        cfg = resolve_path("retrieval_ranker.json")
        try:
            data = {"current": str(model_path), "weights": {}}
            if cfg.exists():
                try:
                    loaded = json.loads(cfg.read_text())
                    if isinstance(loaded, dict):
                        data.update(loaded)
                except Exception:
                    pass
            wmap = {str(k): float(v) for k, v in weights.items()}
            data["current"] = str(model_path)
            data["weights"] = wmap
            tmp = cfg.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(cfg)
        except Exception:
            logger.exception("Failed to persist retrieval ranker config")
        try:
            if weights:
                if hasattr(self.context_builder, "refresh_db_weights"):
                    self.context_builder.refresh_db_weights(weights)  # type: ignore[attr-defined]
                elif hasattr(self.context_builder, "db_weights"):
                    self.context_builder.db_weights.update(weights)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to refresh context builder db weights after reload")

    # ------------------------------------------------------------------
    def reload_reliability_scores(self) -> None:
        """Refresh retriever reliability statistics."""

        try:
            self.retriever.reload_reliability_scores()  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to reload retriever reliability scores")

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
        When ``risk_scores`` are supplied, origins whose vectors closely
        resemble past failures receive a negative penalty proportional to the
        similarity score so risky databases are down-weighted.

        Weights are persisted in :class:`VectorMetricsDB` so the behaviour
        survives restarts.  If the metrics database is unavailable the
        weights are stored only in memory for the lifetime of this instance.
        Setting the ``RANKER_REQUIRE_PERSISTENCE`` environment variable to a
        truthy value will instead raise ``RuntimeError`` when persistence is
        missing.  The new weights are returned as a mapping.
        """

        require_persist = (
            os.getenv("RANKER_REQUIRE_PERSISTENCE", "").lower() in {"1", "true", "yes"}
        )
        if self.vector_metrics is None and require_persist:
            raise RuntimeError("VectorMetricsDB is required for ranker updates")

        origins = {origin or "" for origin, _vec_id, _score in vectors}

        per_db: Dict[str, float] = roi_deltas.copy() if roi_deltas else {}

        # Always consult ROITracker for per-origin deltas so adaptive weights
        # incorporate the latest ROI information.  Tracker-derived deltas only
        # supplement explicit ``roi_deltas`` to avoid double counting.
        if self.roi_tracker is not None:
            try:
                raw = self.roi_tracker.origin_db_deltas()
                for origin in origins:
                    if origin in per_db:
                        continue
                    val = raw.get(origin)
                    if val is not None:
                        per_db[origin] = float(val)
            except Exception:
                logger.exception("Failed to fetch ROI tracker origin deltas")

        if not per_db:
            delta = 0.1 if success else -0.1
            for origin in origins:
                per_db[origin] = per_db.get(origin, 0.0) + delta

        max_penalty = 0.0
        if risk_scores:
            threshold = getattr(
                getattr(self.patch_logger, "patch_safety", None), "threshold", 0.0
            )
            for origin, score in risk_scores.items():
                key = origin or ""
                if threshold and score < threshold:
                    continue
                penalty = abs(score)
                max_penalty = max(max_penalty, penalty)
                per_db[key] = per_db.get(key, 0.0) - penalty
        max_delta = max(abs(d) for d in per_db.values()) if per_db else 0.0

        updates: Dict[str, float] = {}
        if self.vector_metrics is not None:
            for origin, change in per_db.items():
                try:
                    new_wt = self.vector_metrics.update_db_weight(origin, change)
                    updates[origin] = new_wt
                    try:
                        self.vector_metrics.log_ranker_update(
                            origin, delta=change, weight=new_wt
                        )
                    except Exception:
                        logger.exception("Failed to log ranker update for %s", origin)
                except Exception:
                    logger.exception("Failed to update db weight for %s", origin)

            if updates:
                try:
                    all_weights = self.vector_metrics.normalize_db_weights()
                    for origin in list(updates):
                        if origin in all_weights:
                            updates[origin] = all_weights[origin]
                except Exception:
                    try:
                        all_weights = self.vector_metrics.get_db_weights()
                    except Exception:
                        all_weights = updates
            try:
                if hasattr(self.context_builder, "refresh_db_weights"):
                    self.context_builder.refresh_db_weights(
                        all_weights
                    )  # type: ignore[attr-defined]
                elif hasattr(self.context_builder, "db_weights"):
                    try:
                        self.context_builder.db_weights.clear()  # type: ignore[attr-defined]
                        self.context_builder.db_weights.update(
                            all_weights
                        )  # type: ignore[attr-defined]
                    except Exception:
                        self.context_builder.db_weights = dict(
                            all_weights
                        )  # type: ignore[attr-defined]
            except Exception:
                logger.exception(
                    "Failed to apply updated db weights to context builder"
                )

                try:
                    self.vector_metrics.set_db_weights(all_weights)
                except Exception:
                    logger.exception("Failed to persist updated db weights")

                # Persist full weight mapping so external services can reload after
                # restarts.  We merge into ``retrieval_ranker.json`` which already
                # tracks the current ranking model path.
                try:
                    cfg = resolve_path("retrieval_ranker.json")
                    data = {"weights": {}}
                    if cfg.exists():
                        try:
                            loaded = json.loads(cfg.read_text())
                            if isinstance(loaded, dict):
                                data.update(loaded)
                        except Exception:
                            logger.exception("Failed to read retrieval_ranker.json")
                    weights = data.get("weights") or {}
                    if not isinstance(weights, dict):
                        weights = {}
                    for db, wt in all_weights.items():
                        try:
                            weights[str(db)] = float(wt)
                        except Exception:
                            continue
                    data["weights"] = weights
                    tmp = cfg.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data))
                    tmp.replace(cfg)
                except Exception:
                    logger.exception("Failed to persist retrieval ranker weights")
            else:  # ensure context builder picks up any external changes
                try:
                    self.context_builder.refresh_db_weights(
                        vector_metrics=self.vector_metrics
                    )  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("Failed to refresh db weights on context builder")
        else:
            store = getattr(self, "_db_weights", {})
            for origin, change in per_db.items():
                new_wt = store.get(origin, 0.0) + change
                store[origin] = new_wt
                updates[origin] = new_wt
            if updates:
                total = sum(abs(w) for w in store.values()) or 1.0
                all_weights = {k: v / total for k, v in store.items()}
                updates = {k: all_weights[k] for k in updates}
                self._db_weights = all_weights
                try:
                    if hasattr(self.context_builder, "refresh_db_weights"):
                        self.context_builder.refresh_db_weights(
                            all_weights
                        )  # type: ignore[attr-defined]
                    elif hasattr(self.context_builder, "db_weights"):
                        try:
                            self.context_builder.db_weights.clear()  # type: ignore[attr-defined]
                            self.context_builder.db_weights.update(
                                all_weights
                            )  # type: ignore[attr-defined]
                        except Exception:
                            self.context_builder.db_weights = dict(
                                all_weights
                            )  # type: ignore[attr-defined]
                except Exception:
                    logger.exception(
                        "Failed to apply updated db weights to context builder"
                    )

        if per_db:
            bus = self.event_bus
            if bus is not None:
                risks = risk_scores or {}
                for origin, delta in per_db.items():
                    payload = {
                        "db": origin,
                        "roi": delta,
                        "win": success,
                        "risk": float(risks.get(origin, 0.0)),
                    }
                    if origin in updates:
                        payload["weight"] = updates[origin]
                    try:
                        bus.publish("retrieval:feedback", payload)
                    except Exception:
                        logger.exception("Failed to publish retrieval feedback")

        try:
            roi_thresh = float(os.getenv("RANKER_SCHEDULER_ROI_THRESHOLD", "0") or 0.0)
        except Exception:
            roi_thresh = 0.0
        try:
            risk_thresh = float(os.getenv("RANKER_SCHEDULER_RISK_THRESHOLD", "0") or 0.0)
        except Exception:
            risk_thresh = 0.0
        if (
            (roi_thresh and abs(max_delta) >= roi_thresh)
            or (risk_thresh and max_penalty >= risk_thresh)
        ):
            try:
                from analytics import ranker_scheduler as rs

                rs.schedule_retrain([self])
            except Exception:
                logger.exception("Failed to schedule ranker retrain")
        return updates

    # ------------------------------------------------------------------
    @log_and_measure
    def query(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        session_id: str = "",
        prioritise: str | None = None,
    ) -> Tuple[str, str]:
        """Retrieve context for *prompt* and store vector metrics.

        Returns a tuple of ``(context, session_id)``.  The ``session_id`` can
        later be passed to :meth:`record_patch_outcome` to update metrics once
        the patch succeeds or fails.
        """

        stats: Dict[str, Any] = {}
        metadata: Dict[str, List[Dict[str, Any]]]
        try:
            kwargs = {
                "top_k": top_k,
                "include_vectors": True,
                "session_id": session_id,
                "return_stats": True,
                "return_metadata": True,
            }
            if prioritise is not None:
                kwargs["prioritise"] = prioritise
            result = self.context_builder.build_context(prompt, **kwargs)
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
                    logger.exception("Failed to log retrieval metrics")
        self._retrieval_meta[sid] = meta_map
        if self.vector_metrics is not None:
            try:
                self.vector_metrics.save_session(sid, vectors, meta_map)
            except Exception:
                logger.exception("Failed to save retrieval session")
        return context, sid

    # ------------------------------------------------------------------
    @log_and_measure
    async def query_async(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        session_id: str = "",
        prioritise: str | None = None,
    ) -> Tuple[str, str]:
        """Asynchronous wrapper for :meth:`query`."""

        stats: Dict[str, Any] = {}
        metadata: Dict[str, List[Dict[str, Any]]]
        try:
            kwargs = {
                "top_k": top_k,
                "include_vectors": True,
                "session_id": session_id,
                "return_stats": True,
                "return_metadata": True,
            }
            if prioritise is not None:
                kwargs["prioritise"] = prioritise
            result = await self.context_builder.build_async(prompt, **kwargs)
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
                    logger.exception("Failed to log retrieval metrics")
        self._retrieval_meta[sid] = meta_map
        if self.vector_metrics is not None:
            try:
                self.vector_metrics.save_session(sid, vectors, meta_map)
            except Exception:
                logger.exception("Failed to save retrieval session")
        return context, sid

    # ------------------------------------------------------------------

    async def _record_patch_outcome_impl(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
        lines_changed: int | None = None,
        tests_passed: bool | None = None,
        enhancement_name: str | None = None,
        start_time: float | None = None,
        timestamp: float | None = None,
        effort_estimate: float | None = None,
        async_mode: bool = False,
    ) -> None:
        """Shared implementation for patch outcome recording."""
        vectors = self._session_vectors.pop(session_id, [])
        meta = self._retrieval_meta.pop(session_id, None)
        if (not vectors or meta is None) and self.vector_metrics is not None:
            try:
                pending = self.vector_metrics.load_sessions()
                stored = pending.get(session_id)
                if stored:
                    vectors, meta = stored
            except Exception:
                logger.exception("Failed to load vectors for session %s", session_id)
        if not vectors:
            return
        # Extract any pre-computed risk scores from retrieval metadata so they
        # can influence ranking weights even if the patch logger cannot
        # reproduce them (for example when failure embeddings are unavailable).
        risk_scores: Dict[str, float] = {}
        if meta:
            for origin, vid, _ in vectors:
                key = f"{origin}:{vid}" if origin else vid
                info = meta.get(key, {}) if isinstance(meta, dict) else {}
                rs = None
                if isinstance(info, dict):
                    rs = info.get("risk_score")
                    if rs is None:
                        inner = info.get("metadata")
                        if isinstance(inner, dict):
                            rs = inner.get("risk_score")
                if rs is not None:
                    try:
                        ok = origin or ""
                        risk_scores[ok] = max(risk_scores.get(ok, 0.0), float(rs))
                    except Exception:
                        pass

        vec_ids = [(f"{o}:{vid}", score) for o, vid, score in vectors]
        callback_scores: Dict[str, float] = {}

        def _cb(scores: Mapping[str, float]) -> None:
            for o, val in scores.items():
                key = o or ""
                try:
                    callback_scores[key] = max(callback_scores.get(key, 0.0), float(val))
                except Exception:
                    callback_scores[key] = max(callback_scores.get(key, 0.0), 0.0)

        kwargs = {
            "patch_id": patch_id,
            "session_id": session_id,
            "contribution": contribution,
            "retrieval_metadata": meta,
            "lines_changed": lines_changed,
            "tests_passed": tests_passed,
            "enhancement_name": enhancement_name,
            "start_time": start_time,
            "end_time": timestamp,
            "effort_estimate": effort_estimate,
        }
        import inspect

        try:
            sig = inspect.signature(self.patch_logger.track_contributors)
            if "risk_callback" in sig.parameters:
                kwargs["risk_callback"] = _cb
        except Exception:
            pass

        if async_mode:
            result = await self.patch_logger.track_contributors_async(
                vec_ids,
                success,
                **kwargs,
            )
        else:
            result = self.patch_logger.track_contributors(
                vec_ids,
                success,
                **kwargs,
            )

        result_scores = dict(result or {})
        for origin, score in result_scores.items():
            key = origin or ""
            risk_scores[key] = max(risk_scores.get(key, 0.0), score)
        for origin, score in callback_scores.items():
            key = origin or ""
            risk_scores[key] = max(risk_scores.get(key, 0.0), score)
        result_roi_deltas = getattr(result, "roi_deltas", None)
        roi_contribs: Dict[str, float] = {}
        roi_actuals: Dict[str, float] = {}
        used_tracker_deltas = False
        roi_drop = False
        if isinstance(result_roi_deltas, Mapping):
            for origin, val in result_roi_deltas.items():
                key = origin or ""
                try:
                    val = float(val)
                except Exception:
                    continue
                roi_actuals[key] = val
                roi_contribs[key] = abs(val)
                roi_drop = roi_drop or val < 0
            if roi_contribs:
                used_tracker_deltas = True

        if not success:
            errors = getattr(result, "errors", []) if result else []
            ps = getattr(self.patch_logger, "patch_safety", None)
            if errors and ps is not None:
                for err in errors:
                    try:
                        ps.record_failure(dict(err))
                    except Exception:
                        logger.exception("Failed to record failure metadata")
            if ps is not None:
                try:
                    ps.load_failures(force=True)
                except Exception:
                    logger.exception("Failed to reload failure vectors")
            builder_ps = getattr(self.context_builder, "patch_safety", None)
            if builder_ps is not None and builder_ps is not ps:
                try:
                    builder_ps.load_failures(force=True)
                except Exception:
                    logger.exception("Failed to reload failure vectors")

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
                if not used_tracker_deltas:
                    deltas = self.roi_tracker.origin_db_deltas()
                    for origin, _vid, _score in vectors:
                        key = origin or ""
                        val = deltas.get(key)
                        if val is not None:
                            val = float(val)
                            roi_actuals[key] = val
                            roi_contribs[key] = abs(val)
                            roi_drop = roi_drop or val < 0
                    if roi_contribs:
                        used_tracker_deltas = True
            except Exception:
                logger.exception("Failed to update ROI tracker with retrieval metrics")

        base = 0.0 if contribution is None else contribution
        for origin, _vid, score in vectors:
            key = origin or ""
            if key in roi_contribs:
                continue
            roi = base if contribution is not None else score
            roi_actuals[key] = roi
            roi_contribs[key] = roi_contribs.get(key, 0.0) + abs(roi)
            if roi < 0:
                roi_drop = True

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
                logger.exception("Failed to update ROI tracker DB metrics")

        roi_deltas: Dict[str, float] = {}
        for origin, roi in roi_actuals.items():
            delta = roi if success else -abs(roi)
            roi_deltas[origin] = delta
            if delta < 0:
                roi_drop = True
        bus = getattr(self.patch_logger, "event_bus", None)
        if bus is None and UnifiedEventBus is not None:
            try:
                bus = UnifiedEventBus()
            except Exception:
                bus = None
        if bus is not None:
            for origin, delta in roi_deltas.items():
                payload = {
                    "db": origin,
                    "roi": delta,
                    "win": success,
                    "risk": float(risk_scores.get(origin, 0.0)),
                }
                try:
                    bus.publish("retrieval:feedback", payload)
                except Exception:
                    logger.exception("Failed to publish retrieval feedback")
        updates = self.update_ranker(
            vectors, success, roi_deltas=roi_deltas, risk_scores=risk_scores
        )
        if updates and hasattr(self.context_builder, "refresh_db_weights"):
            try:
                self.context_builder.refresh_db_weights()
            except Exception:
                logger.exception("Failed to refresh context builder weights")

        if not success or roi_drop:
            try:
                await schedule_backfill(dbs=list(roi_deltas.keys()))
            except Exception:
                logger.exception("Failed to schedule embedding backfill")
            try:
                self.reload_reliability_scores()
            except Exception:
                logger.exception("Failed to reload retriever reliability scores")

        if self.vector_metrics is not None:
            try:
                self.vector_metrics.delete_session(session_id)
            except Exception:
                logger.exception("Failed to delete session %s", session_id)

    # ------------------------------------------------------------------
    def record_patch_outcome(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
        lines_changed: int | None = None,
        tests_passed: bool | None = None,
        enhancement_name: str | None = None,
        start_time: float | None = None,
        timestamp: float | None = None,
        effort_estimate: float | None = None,
    ) -> None:
        """Forward patch outcome to :class:`PatchLogger`."""

        asyncio.run(
            self._record_patch_outcome_impl(
                session_id,
                success,
                patch_id=patch_id,
                contribution=contribution,
                lines_changed=lines_changed,
                tests_passed=tests_passed,
                enhancement_name=enhancement_name,
                start_time=start_time,
                timestamp=timestamp,
                effort_estimate=effort_estimate,
                async_mode=False,
            )
        )

    # ------------------------------------------------------------------
    async def record_patch_outcome_async(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
        lines_changed: int | None = None,
        tests_passed: bool | None = None,
        enhancement_name: str | None = None,
        start_time: float | None = None,
        timestamp: float | None = None,
        effort_estimate: float | None = None,
    ) -> None:
        """Asynchronous wrapper for :meth:`record_patch_outcome`."""

        await self._record_patch_outcome_impl(
            session_id,
            success,
            patch_id=patch_id,
            contribution=contribution,
            lines_changed=lines_changed,
            tests_passed=tests_passed,
            enhancement_name=enhancement_name,
            start_time=start_time,
            timestamp=timestamp,
            effort_estimate=effort_estimate,
            async_mode=True,
        )

    # ------------------------------------------------------------------
    def roi_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return latest ROI statistics grouped by origin type.

        The method aggregates metrics from :class:`vector_metrics_db.VectorMetricsDB`
        using :func:`analytics.session_roi.origin_roi`.  An empty mapping is
        returned when metrics are unavailable or aggregation fails.
        """

        if self.vector_metrics is None:
            return {}
        try:
            from analytics.session_roi import origin_roi

            return origin_roi(self.vector_metrics)
        except Exception:
            logger.exception("Failed to retrieve ROI stats")
            return {}


__all__ = ["CognitionLayer"]
