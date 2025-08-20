from __future__ import annotations

"""Helper for recording patch outcomes for contributing vectors."""

from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

import asyncio
import time

from .decorators import log_and_measure

try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    import metrics_exporter as _me  # type: ignore

_TRACK_OUTCOME = _me.Gauge(
    "patch_logger_track_contributors_total",
    "Outcomes recorded by PatchLogger.track_contributors",
    labelnames=["status"],
)
_TRACK_DURATION = _me.Gauge(
    "patch_logger_track_contributors_duration_seconds",
    "Duration of PatchLogger.track_contributors calls",
)

_VECTOR_RISK = _me.Gauge(
    "patch_logger_vectors_total",
    "Vectors processed by PatchLogger grouped by risk level",
    labelnames=["risk"],
)

try:  # pragma: no cover - optional dependencies
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

try:  # pragma: no cover
    from code_database import PatchHistoryDB  # type: ignore
except Exception:  # pragma: no cover
    PatchHistoryDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover
    UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover
    ROITracker = None  # type: ignore


class PatchLogger:
    """Record patch outcomes in ``PatchHistoryDB`` or ``VectorMetricsDB``.

    ``metrics_db`` is accepted for backwards compatibility and maps to the
    older :class:`data_bot.MetricsDB` API.  When provided it takes precedence
    over the more granular databases.
    """

    def __init__(
        self,
        patch_db: PatchHistoryDB | None = None,
        vector_metrics: VectorMetricsDB | None = None,
        metrics_db: Any | None = None,
        roi_tracker: ROITracker | None = None,
        event_bus: "UnifiedEventBus" | None = None,
    ) -> None:
        self.patch_db = patch_db or (PatchHistoryDB() if PatchHistoryDB is not None else None)
        self.vector_metrics = vector_metrics or (
            VectorMetricsDB() if VectorMetricsDB is not None else None
        )
        self.metrics_db = metrics_db
        self.roi_tracker = roi_tracker
        self.event_bus = event_bus

    # ------------------------------------------------------------------
    def _parse_vectors(
        self,
        vector_ids: Union[
            Iterable[Union[str, Tuple[str, float]]],
            Mapping[str, float],
        ],
    ) -> List[Tuple[str, str, float]]:
        pairs: List[Tuple[str, str, float]] = []
        items: Iterable[Union[str, Tuple[str, float]]]
        if isinstance(vector_ids, Mapping):
            items = vector_ids.items()  # type: ignore[assignment]
        else:
            items = vector_ids
        for item in items:
            if isinstance(item, tuple):
                vid, score = item
            else:
                vid, score = item, 0.0
            if ":" in vid:
                origin, vec_id = vid.split(":", 1)
            else:
                origin, vec_id = "", vid
            pairs.append((origin, vec_id, float(score)))
        return pairs

    # ------------------------------------------------------------------
    @log_and_measure
    def track_contributors(
        self,
        vector_ids: Union[Mapping[str, float], Sequence[Union[str, Tuple[str, float]]]],
        result: bool,
        *,
        patch_id: str = "",
        session_id: str = "",
        contribution: float | None = None,
        retrieval_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Log patch outcome for vectors contributing to a patch."""

        start = time.time()
        status = "success" if result else "failure"
        try:
            detailed = self._parse_vectors(vector_ids)
            detailed.sort(key=lambda t: t[2], reverse=True)
            pairs = [(o, vid) for o, vid, _ in detailed]
            meta = retrieval_metadata or {}
            detailed_meta = []
            provenance_meta = []
            vm_vectors = []
            risky = 0
            safe = 0
            for o, vid, score in detailed:
                key = f"{o}:{vid}" if o else vid
                m = meta.get(key, {})
                lic = m.get("license")
                fp = m.get("license_fingerprint")
                alerts = m.get("semantic_alerts")
                sev = m.get("alignment_severity")
                if fp is not None:
                    detailed_meta.append((o, vid, score, lic, fp, alerts))
                    provenance_meta.append((o, vid, score, lic, fp, alerts, sev))
                else:
                    detailed_meta.append((o, vid, score, lic, alerts))
                    provenance_meta.append((o, vid, score, lic, alerts, sev))
                vm_vectors.append((vid, score, lic, alerts, sev))
                if sev:
                    risky += 1
                else:
                    safe += 1

            _VECTOR_RISK.labels("risky").inc(risky)
            _VECTOR_RISK.labels("safe").inc(safe)

            if self.metrics_db is not None:
                try:  # pragma: no cover - legacy path
                    self.metrics_db.log_patch_outcome(
                        patch_id or "", result, pairs, session_id=session_id
                    )
                except Exception:
                    pass
            else:
                if self.patch_db is not None and patch_id:
                    try:  # pragma: no cover - best effort
                        self.patch_db.record_vector_metrics(
                            session_id,
                            pairs,
                            patch_id=int(patch_id),
                            contribution=0.0 if contribution is None else contribution,
                            win=result,
                            regret=not result,
                        )
                    except Exception:
                        pass
                    try:
                        self.patch_db.record_provenance(int(patch_id), provenance_meta)
                    except Exception:
                        pass
                    try:
                        self.patch_db.log_ancestry(int(patch_id), detailed_meta)
                    except Exception:
                        pass
                    try:
                        self.patch_db.log_contributors(
                            int(patch_id), detailed, session_id
                        )
                    except Exception:
                        pass
                roi_base = 0.0 if contribution is None else contribution
                origin_totals: dict[str, float] = {}
                for origin, vid, score in detailed:
                    roi = roi_base if contribution is not None else score
                    key = origin or ""
                    origin_totals[key] = origin_totals.get(key, 0.0) + roi
                    if self.vector_metrics is not None:
                        try:  # pragma: no cover - best effort
                            self.vector_metrics.update_outcome(
                                session_id,
                                [(origin, vid)],
                                contribution=roi,
                                patch_id=patch_id,
                                win=result,
                                regret=not result,
                            )
                        except Exception:
                            pass
                if self.vector_metrics is not None:
                    for origin, roi in origin_totals.items():
                        try:
                            self.vector_metrics.log_retrieval_feedback(
                                origin, win=result, regret=not result, roi=roi
                            )
                        except Exception:
                            pass
                        if self.event_bus is not None:
                            try:
                                self.event_bus.publish(
                                    "retrieval:feedback",
                                    {"db": origin, "win": result, "regret": not result},
                                )
                            except Exception:
                                pass
                        elif UnifiedEventBus is not None:
                            try:
                                UnifiedEventBus().publish(
                                    "retrieval:feedback",
                                    {"db": origin, "win": result, "regret": not result},
                                )
                            except Exception:
                                pass
                    if patch_id:
                        try:
                            self.vector_metrics.record_patch_ancestry(patch_id, vm_vectors)
                        except Exception:
                            pass
                if origin_totals:
                    metrics = {
                        origin: {
                            "roi": roi,
                            "win_rate": 1.0 if result else 0.0,
                            "regret_rate": 0.0 if result else 1.0,
                        }
                        for origin, roi in origin_totals.items()
                    }
                    if self.roi_tracker is not None:
                        try:
                            self.roi_tracker.update_db_metrics(metrics)
                        except Exception:
                            pass
                    else:
                        for origin, stats in metrics.items():
                            payload = {"db": origin, **stats}
                            if self.event_bus is not None:
                                try:
                                    self.event_bus.publish("roi:update", payload)
                                except Exception:
                                    pass
                            elif UnifiedEventBus is not None:
                                try:
                                    UnifiedEventBus().publish("roi:update", payload)
                                except Exception:
                                    pass
                unique_origins = {o for o, _, _ in detailed if o}
                if unique_origins:
                    if self.event_bus is not None:
                        for origin in unique_origins:
                            try:
                                self.event_bus.publish(
                                    "embedding:backfill", {"db": origin}
                                )
                            except Exception:
                                pass
                    elif UnifiedEventBus is not None:
                        bus = UnifiedEventBus()
                        for origin in unique_origins:
                            try:
                                bus.publish("embedding:backfill", {"db": origin})
                            except Exception:
                                pass
        except Exception:
            _TRACK_OUTCOME.labels("error").inc()
            _TRACK_DURATION.set(time.time() - start)
            raise

        _TRACK_OUTCOME.labels(status).inc()
        _TRACK_DURATION.set(time.time() - start)

        if self.event_bus is not None:
            try:
                self.event_bus.publish("patch_logger:outcome", {"result": result})
            except Exception:
                pass
        elif UnifiedEventBus is not None:
            try:
                UnifiedEventBus().publish("patch_logger:outcome", {"result": result})
            except Exception:
                pass

    # ------------------------------------------------------------------
    @log_and_measure
    async def track_contributors_async(
        self,
        vector_ids: Union[Mapping[str, float], Sequence[Union[str, Tuple[str, float]]]],
        result: bool,
        *,
        patch_id: str = "",
        session_id: str = "",
        contribution: float | None = None,
        retrieval_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Asynchronous wrapper for :meth:`track_contributors`."""

        await asyncio.to_thread(
            self.track_contributors.__wrapped__,
            self,
            vector_ids,
            result,
            patch_id=patch_id,
            session_id=session_id,
            contribution=contribution,
            retrieval_metadata=retrieval_metadata,
        )


__all__ = ["PatchLogger"]

