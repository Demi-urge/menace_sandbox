from __future__ import annotations

"""Helper for recording patch outcomes for contributing vectors."""

from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

import asyncio
import time

from .decorators import log_and_measure
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from .patch_safety import check_patch_safety

try:  # pragma: no cover - optional dependency for similarity risk
    from patch_safety import PatchSafety  # type: ignore
except Exception:  # pragma: no cover
    PatchSafety = None  # type: ignore

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

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())

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

try:  # pragma: no cover - optional summariser
    from menace_memory_manager import _summarise_text  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback
    def _summarise_text(text: str, ratio: float = 0.2) -> str:
        text = text.strip()
        if len(text) <= 120:
            return text
        return text[:117] + "..."

try:  # pragma: no cover - optional info database
    from research_aggregator_bot import InfoDB, ResearchItem  # type: ignore
except Exception:  # pragma: no cover
    InfoDB = ResearchItem = None  # type: ignore

try:  # pragma: no cover - optional enhancement database
    from chatgpt_enhancement_bot import EnhancementDB, Enhancement  # type: ignore
except Exception:  # pragma: no cover
    EnhancementDB = Enhancement = None  # type: ignore


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
        info_db: InfoDB | None = None,
        enhancement_db: EnhancementDB | None = None,
        max_alert_severity: float = 1.0,
        max_alerts: int = 5,
        license_denylist: set[str] | None = None,
        patch_safety: PatchSafety | None = None,
    ) -> None:
        self.patch_db = patch_db or (PatchHistoryDB() if PatchHistoryDB is not None else None)
        self.vector_metrics = vector_metrics or (
            VectorMetricsDB() if VectorMetricsDB is not None else None
        )
        self.metrics_db = metrics_db
        self.roi_tracker = roi_tracker
        self.event_bus = event_bus
        self.info_db = info_db
        self.enhancement_db = enhancement_db
        self.max_alert_severity = max_alert_severity
        self.max_alerts = max_alerts
        self.license_denylist = set(license_denylist or _DEFAULT_LICENSE_DENYLIST)
        self.patch_safety = patch_safety or (PatchSafety() if PatchSafety is not None else None)

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
    ) -> dict[str, float]:
        """Log patch outcome for vectors contributing to a patch.

        Returns a mapping of origin database to the maximum risk score
        calculated by :class:`patch_safety.PatchSafety` for the vectors that
        contributed to the patch.
        """

        start = time.time()
        status = "success" if result else "failure"
        roi_metrics: dict[str, dict[str, Any]] = {}
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
            filtered = 0
            origin_alerts: dict[str, set[str]] = {}
            origin_sev: dict[str, float] = {}
            origin_risk: dict[str, float] = {}
            for o, vid, score in detailed:
                key = f"{o}:{vid}" if o else vid
                m = meta.get(key, {})
                if not check_patch_safety(
                    m,
                    max_alert_severity=self.max_alert_severity,
                    max_alerts=self.max_alerts,
                    license_denylist=self.license_denylist,
                ):
                    filtered += 1
                    continue
                lic = m.get("license")
                fp = m.get("license_fingerprint")
                alerts = m.get("semantic_alerts")
                sev = m.get("alignment_severity")
                risk = 0.0
                if self.patch_safety is not None:
                    try:
                        risk = float(self.patch_safety.score(m))
                    except Exception:
                        risk = 0.0
                if self.patch_safety is not None and risk >= getattr(self.patch_safety, "threshold", 1.0):
                    payload = {"vector": key, "score": risk, "threshold": self.patch_safety.threshold}
                    bus = self.event_bus
                    if bus is None and UnifiedEventBus is not None:
                        try:
                            bus = UnifiedEventBus()
                        except Exception:
                            bus = None
                    if bus is not None:
                        try:
                            bus.publish("patch:risk_alert", payload)
                        except Exception:
                            pass
                if fp is not None:
                    detailed_meta.append((o, vid, score, lic, fp, alerts))
                    provenance_meta.append((o, vid, score, lic, fp, alerts, sev))
                else:
                    detailed_meta.append((o, vid, score, lic, alerts))
                    provenance_meta.append((o, vid, score, lic, alerts, sev))
                vm_vectors.append((vid, score, lic, alerts, sev, risk))
                ok = o or ""
                origin_risk[ok] = max(origin_risk.get(ok, 0.0), risk)
                if sev:
                    try:
                        origin_sev[ok] = max(origin_sev.get(ok, 0.0), float(sev))
                    except Exception:
                        origin_sev[ok] = max(origin_sev.get(ok, 0.0), 1.0)
                if self.patch_safety is not None and risk >= getattr(self.patch_safety, "threshold", 1.0):
                    risky += 1
                else:
                    safe += 1
                if alerts:
                    if isinstance(alerts, (list, tuple, set)):
                        origin_alerts.setdefault(ok, set()).update(map(str, alerts))
                    else:
                        origin_alerts.setdefault(ok, set()).add(str(alerts))

            _VECTOR_RISK.labels("risky").inc(risky)
            _VECTOR_RISK.labels("safe").inc(safe)
            if filtered:
                _VECTOR_RISK.labels("filtered").inc(filtered)

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
                            pairs,
                            contribution=roi_base if contribution is not None else 0.0,
                            patch_id=patch_id,
                            win=result,
                            regret=not result,
                        )
                    except Exception:
                        pass
                    if patch_id:
                        try:
                            self.vector_metrics.record_patch_ancestry(patch_id, vm_vectors)
                        except Exception:
                            pass
                for origin, roi in origin_totals.items():
                    if self.vector_metrics is not None:
                        try:
                            self.vector_metrics.log_retrieval_feedback(
                                origin, win=result, regret=not result, roi=roi
                            )
                        except Exception:
                            pass
                    payload = {
                        "db": origin,
                        "win": result,
                        "regret": not result,
                        "roi": roi,
                        "win_rate": 1.0 if result else 0.0,
                        "regret_rate": 0.0 if result else 1.0,
                    }
                    if self.event_bus is not None:
                        try:
                            self.event_bus.publish("retrieval:feedback", payload)
                        except Exception:
                            pass
                    elif UnifiedEventBus is not None:
                        try:
                            UnifiedEventBus().publish("retrieval:feedback", payload)
                        except Exception:
                            pass
                roi_metrics: dict[str, dict[str, Any]] = {}
                if origin_totals:
                    for origin, roi in origin_totals.items():
                        metrics = {
                            "roi": roi,
                            "win_rate": 1.0 if result else 0.0,
                            "regret_rate": 0.0 if result else 1.0,
                            "alignment_severity": origin_sev.get(origin, 0.0),
                            "semantic_alerts": sorted(origin_alerts.get(origin, [])),
                            "risk_score": origin_risk.get(origin, 0.0),
                        }
                        roi_metrics[origin] = metrics
                    if self.roi_tracker is not None:
                        for origin, stats in roi_metrics.items():
                            try:
                                # send deltas for each origin individually
                                self.roi_tracker.update_db_metrics({origin: stats})
                            except Exception:
                                pass
                    else:
                        for origin, stats in roi_metrics.items():
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
                if result and retrieval_metadata:
                    for origin, vid, _ in detailed:
                        key = f"{origin}:{vid}" if origin else vid
                        meta = retrieval_metadata.get(key, {})
                        text = ""
                        if origin == "code":
                            text = meta.get("code") or meta.get("summary") or ""
                        elif origin == "error":
                            text = meta.get("message") or meta.get("description") or ""
                        elif origin == "enhancement":
                            text = (
                                meta.get("description")
                                or meta.get("lessons")
                                or meta.get("title")
                                or meta.get("idea")
                                or ""
                            )
                        elif origin == "information":
                            text = meta.get("content") or meta.get("summary") or ""
                        if not text:
                            continue
                        try:
                            summary = _summarise_text(str(text))
                        except Exception:
                            summary = str(text)[:120]
                        try:
                            if (
                                origin == "enhancement"
                                and self.enhancement_db is not None
                                and Enhancement is not None
                            ):
                                enh = Enhancement(
                                    idea=summary,
                                    rationale="",
                                    summary=summary,
                                    description=summary,
                                )
                                self.enhancement_db.add(enh)
                            elif (
                                self.info_db is not None
                                and InfoDB is not None
                                and ResearchItem is not None
                            ):
                                item = ResearchItem(
                                    topic=f"patch:{patch_id}" if patch_id else "patch",
                                    content=summary,
                                    summary=summary,
                                    category=origin,
                                    type_="lesson",
                                )
                                self.info_db.add(item)
                        except Exception:
                            pass
        except Exception:
            _TRACK_OUTCOME.labels("error").inc()
            _TRACK_DURATION.set(time.time() - start)
            raise

        _TRACK_OUTCOME.labels(status).inc()
        _TRACK_DURATION.set(time.time() - start)

        all_alerts: list[str] = []
        if origin_alerts:
            all_alerts = sorted({a for s in origin_alerts.values() for a in s})
        max_sev = max(origin_sev.values(), default=0.0)
        max_risk = max(origin_risk.values(), default=0.0)
        payload = {
            "result": result,
            "roi_metrics": roi_metrics,
            "win": result,
            "regret": not result,
            "alignment_severity": max_sev,
            "semantic_alerts": all_alerts,
            "risk_score": max_risk,
        }
        if patch_id:
            payload["patch_id"] = patch_id

        if self.event_bus is not None:
            try:
                self.event_bus.publish("patch_logger:outcome", payload)
            except Exception:
                pass
        elif UnifiedEventBus is not None:
            try:
                UnifiedEventBus().publish("patch_logger:outcome", payload)
            except Exception:
                pass

        return origin_risk

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
    ) -> dict[str, float]:
        """Asynchronous wrapper for :meth:`track_contributors`."""

        return await asyncio.to_thread(
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

