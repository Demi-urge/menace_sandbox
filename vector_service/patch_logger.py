from __future__ import annotations

"""Helper for recording patch outcomes for contributing vectors."""

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import asyncio
import logging
import time
import json

from .decorators import log_and_measure
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from patch_safety import PatchSafety
from db_router import GLOBAL_ROUTER, init_db_router
from .weight_adjuster import WeightAdjuster
from .roi_tags import RoiTag
from enhancement_score import (
    EnhancementMetrics,
    compute_enhancement_score as _compute_enhancement_score,
)

if TYPE_CHECKING:  # pragma: no cover
    from .vectorizer import SharedVectorService

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

_TRACK_FAILURES = _me.Gauge(
    "patch_logger_failures_total",
    "Errors recorded by PatchLogger.track_contributors",
)

_TRACK_TESTS = _me.Gauge(
    "patch_logger_tests_total",
    "Counts of test results recorded by PatchLogger.track_contributors",
    labelnames=["status"],
)

# Per-database metrics captured for Prometheus dashboards.  Gauges are
# defined lazily to avoid duplicate registration when modules are reloaded.
try:  # pragma: no cover - metrics optional
    _DB_RISK = _me.Gauge(
        "patch_logger_db_risk_score",
        "Risk score per origin database from the latest patch",
        ["origin_db"],
    )
    _DB_ROI_DELTA = _me.Gauge(
        "patch_logger_db_roi_delta",
        "ROI delta per origin database from the latest patch",
        ["origin_db"],
    )
except Exception:  # pragma: no cover - gauges may already exist
    try:
        from prometheus_client import REGISTRY  # type: ignore

        _DB_RISK = REGISTRY._names_to_collectors.get(  # type: ignore[attr-defined]
            "patch_logger_db_risk_score"
        )
        _DB_ROI_DELTA = REGISTRY._names_to_collectors.get(
            "patch_logger_db_roi_delta"
        )
    except Exception:  # pragma: no cover - metrics unavailable
        _DB_RISK = None
        _DB_ROI_DELTA = None

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

try:  # pragma: no cover - optional patch score logging
    from patch_score_backend import _log_outcome as _ps_log_outcome  # type: ignore
except Exception:  # pragma: no cover
    _ps_log_outcome = None  # type: ignore

# Restricted set of ROI tags used to annotate patch outcomes is defined in
# ``roi_tags.RoiTag``.

try:  # pragma: no cover - optional precise tokenizer
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - dependency missing
    tiktoken = None  # type: ignore
    _ENCODER = None

try:  # pragma: no cover - optional summariser
    from menace_memory_manager import _summarise_text  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback
    try:  # pragma: no cover - local summarisation model
        from gensim.summarization import summarize as _gs  # type: ignore
    except Exception:  # pragma: no cover - dependency missing
        _gs = None  # type: ignore

    _MAX_INPUT_TOKENS = 2048
    _MAX_SUMMARY_TOKENS = 128

    def _trim_to_tokens(text: str, limit: int) -> str:
        if _ENCODER is not None:
            tokens = _ENCODER.encode(text)
            if len(tokens) > limit:
                return _ENCODER.decode(tokens[:limit])
            return text
        if len(text) > limit * 4:
            return text[: limit * 4]
        return text

    def _summarise_text(text: str, ratio: float = 0.2) -> str:
        text = text.strip().replace("\n", " ")
        if not text:
            return ""
        text = _trim_to_tokens(text, _MAX_INPUT_TOKENS)
        summary = ""
        if _gs is not None:
            try:
                summary = _gs(text, ratio=ratio)
            except Exception:
                summary = ""
        if not summary:
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if not sentences:
                summary = text
            else:
                count = max(1, int(len(sentences) * ratio))
                summary = ". ".join(sentences[:count]) + "."
        return _trim_to_tokens(summary, _MAX_SUMMARY_TOKENS)

try:  # pragma: no cover - optional info database
    from research_aggregator_bot import InfoDB, ResearchItem  # type: ignore
except Exception:  # pragma: no cover
    InfoDB = ResearchItem = None  # type: ignore

try:  # pragma: no cover - optional enhancement database
    from chatgpt_enhancement_bot import EnhancementDB, Enhancement  # type: ignore
except Exception:  # pragma: no cover
    EnhancementDB = Enhancement = None  # type: ignore


logger = logging.getLogger(__name__)
router = GLOBAL_ROUTER or init_db_router("patch_logger")


class TrackResult(dict):
    """Mapping of origin similarity scores with attached error metadata."""

    def __init__(
        self,
        mapping: Mapping[str, float] | None = None,
        *,
        errors=None,
        tests_passed: bool | None = None,
        lines_changed: int | None = None,
        context_tokens: int | None = None,
        patch_difficulty: int | None = None,
        duration_s: float | None = None,
        error_count: int | None = None,
        effort_estimate: float | None = None,
        roi_deltas: Mapping[str, float] | None = None,
        roi_delta: float | None = None,
        enhancement_score: float | None = None,
    ) -> None:
        super().__init__(mapping or {})
        self.errors = list(errors or [])
        self.tests_passed = tests_passed
        self.lines_changed = lines_changed
        self.context_tokens = context_tokens
        self.patch_difficulty = patch_difficulty
        self.duration_s = duration_s
        self.error_count = error_count
        self.effort_estimate = effort_estimate
        self.human_effort = effort_estimate
        self.enhancement_score = enhancement_score
        # ``roi_deltas`` allows callers to retrieve per-origin ROI changes
        # computed during :meth:`track_contributors`.  It defaults to an empty
        # mapping for backwards compatibility so existing callers treating the
        # result as a plain ``dict`` continue to work without modification.
        self.roi_deltas = dict(roi_deltas or {})
        self.roi_delta = roi_delta


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
        vector_service: SharedVectorService | None = None,
        info_db: InfoDB | None = None,
        enhancement_db: EnhancementDB | None = None,
        max_alert_severity: float = 1.0,
        max_alerts: int = 5,
        license_denylist: set[str] | None = None,
        patch_safety: PatchSafety | None = None,
        weight_adjuster: WeightAdjuster | None = None,
        bootstrap_fast: bool = False,
    ) -> None:
        self.bootstrap_fast = bool(bootstrap_fast)
        if patch_db is not None:
            self.patch_db = patch_db
        elif PatchHistoryDB is not None:
            try:
                self.patch_db = PatchHistoryDB(
                    bootstrap=self.bootstrap_fast,
                    bootstrap_fast=self.bootstrap_fast,
                )
            except Exception:
                self.patch_db = None
        else:
            self.patch_db = None
        if vector_metrics is not None:
            self.vector_metrics = vector_metrics
        elif VectorMetricsDB is not None:
            try:
                self.vector_metrics = VectorMetricsDB(
                    bootstrap_fast=self.bootstrap_fast
                )
            except Exception:
                self.vector_metrics = None
        else:
            self.vector_metrics = None
        self.metrics_db = metrics_db
        self.roi_tracker = roi_tracker
        self.event_bus = event_bus
        self.vector_service = vector_service
        self.info_db = info_db
        self.enhancement_db = enhancement_db
        self.max_alert_severity = max_alert_severity
        self.max_alerts = max_alerts
        self.license_denylist = set(license_denylist or _DEFAULT_LICENSE_DENYLIST)
        self.patch_safety = patch_safety or PatchSafety()
        self.patch_safety.max_alert_severity = max_alert_severity
        self.patch_safety.max_alerts = max_alerts
        self.patch_safety.license_denylist = self.license_denylist
        self.weight_adjuster = weight_adjuster or (
            WeightAdjuster(vector_metrics=self.vector_metrics)
            if self.vector_metrics is not None
            else None
        )
        if self.bootstrap_fast:
            logger.info(
                "patch_logger.bootstrap_fast.enabled; skipping heavy patch history schema setup",
                extra={
                    "bootstrap_fast": True,
                    "patch_db": type(self.patch_db).__name__ if self.patch_db else None,
                    "vector_metrics": type(self.vector_metrics).__name__
                    if self.vector_metrics
                    else None,
                },
            )

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
        roi_delta: float | None = None,
        retrieval_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        risk_callback: "Callable[[Mapping[str, float]], Any]" | None = None,
        lines_changed: int | None = None,
        tests_passed: bool | None = None,
        enhancement_name: str | None = None,
        context_tokens: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        diff: str | None = None,
        summary: str | None = None,
        outcome: str | None = None,
        error_summary: str | None = None,
        error_traces: Sequence[Mapping[str, Any]] | None = None,
        roi_tag: RoiTag | str | None = None,
        effort_estimate: float | None = None,
        risk_flags: Sequence[str] | None = None,
        diff_risk_score: float | None = None,
    ) -> dict[str, float]:
        """Log patch outcome for vectors contributing to a patch.

        Returns a mapping of origin database to the maximum similarity score
        calculated by :class:`patch_safety.PatchSafety` for the vectors that
        contributed to the patch.  Higher scores indicate a closer match to
        previously recorded failures and can be used by callers to penalise
        risky origins.
        """

        start = time.time()
        status = "success" if result else "failure"
        if contribution is None and roi_delta is not None:
            contribution = roi_delta
        roi_tag_val = RoiTag.validate(roi_tag)
        roi_metrics: dict[str, dict[str, Any]] = {}
        roi_deltas: dict[str, float] = {}
        context_tokens_val = int(context_tokens or 0)
        patch_difficulty = (lines_changed or 0)
        effort_estimate_val = effort_estimate
        time_to_completion = (
            None
            if start_time is None or end_time is None
            else float(end_time) - float(start_time)
        )
        try:
            detailed = self._parse_vectors(vector_ids)
            detailed.sort(key=lambda t: t[2], reverse=True)
            pairs = [(o, vid) for o, vid, _ in detailed]
            meta = retrieval_metadata or {}
            if context_tokens is None:
                for m in meta.values():
                    if isinstance(m, Mapping):
                        try:
                            context_tokens_val += int(m.get("prompt_tokens") or 0)
                        except Exception:
                            pass
            patch_difficulty = (lines_changed or 0) + context_tokens_val
            errors: list[Mapping[str, Any]] = list(error_traces or [])
            if error_summary:
                errors.append({"summary": error_summary})
            if not result and meta:
                generic_error = meta.get("error")
                if generic_error:
                    errors.append(generic_error)
            summary_arg = summary if summary is not None else error_summary
            detailed_meta = []
            provenance_meta = []
            vm_vectors = []
            risky = 0
            safe = 0
            filtered = 0
            origin_alerts: dict[str, set[str]] = {}
            origin_sev: dict[str, float] = {}
            origin_similarity: dict[str, float] = {}
            for o, vid, score in detailed:
                key = f"{o}:{vid}" if o else vid
                m = meta.get(key, {})
                self.patch_safety.max_alert_severity = self.max_alert_severity
                self.patch_safety.max_alerts = self.max_alerts
                self.patch_safety.license_denylist = self.license_denylist
                passed, similarity, risks = self.patch_safety.evaluate(m, m, origin=o)
                if not result:
                    try:
                        self.patch_safety.record_failure(m, origin=o)
                    except Exception:
                        logger.exception("Failed to record failure metadata")
                if similarity >= self.patch_safety.threshold:
                    payload = {
                        "vector": key,
                        "score": similarity,
                        "threshold": self.patch_safety.threshold,
                    }
                    bus = self.event_bus
                    if bus is None and UnifiedEventBus is not None:
                        try:
                            bus = UnifiedEventBus()
                        except Exception:
                            logger.exception("Failed to create UnifiedEventBus for risk alerts")
                            bus = None
                    if bus is not None:
                        try:
                            bus.publish("patch:risk_alert", payload)
                        except Exception:
                            logger.exception("Failed to publish patch risk alert")
                ok = o or ""
                vector_risk = max(risks.get(ok, 0.0), similarity)
                origin_similarity[ok] = max(origin_similarity.get(ok, 0.0), vector_risk)
                if not passed:
                    lic = m.get("license")
                    alerts = m.get("semantic_alerts")
                    fp = m.get("license_fingerprint")
                    sev = m.get("alignment_severity")
                    blocked = (
                        lic in self.license_denylist
                        or fp in self.license_denylist
                        or alerts
                        or (
                            sev is not None
                            and self.max_alert_severity is not None
                            and float(sev) > float(self.max_alert_severity)
                        )
                    )
                    if blocked:
                        origin_similarity.pop(ok, None)
                    filtered += 1
                    continue
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
                vm_vectors.append((vid, score, lic, alerts, sev, vector_risk))
                if sev:
                    try:
                        origin_sev[ok] = max(origin_sev.get(ok, 0.0), float(sev))
                    except Exception:
                        origin_sev[ok] = max(origin_sev.get(ok, 0.0), 1.0)
                if similarity >= self.patch_safety.threshold:
                    risky += 1
                else:
                    safe += 1
                if alerts:
                    if isinstance(alerts, (list, tuple, set)):
                        origin_alerts.setdefault(ok, set()).update(map(str, alerts))
                    else:
                        origin_alerts.setdefault(ok, set()).add(str(alerts))

                if not result:
                    err = None
                    if o == "error":
                        err = m or {}
                    else:
                        err = m.get("error") if isinstance(m, Mapping) else None
                    if err:
                        try:
                            errors.append(dict(err))
                        except Exception:
                            errors.append({})

            _VECTOR_RISK.labels("risky").inc(risky)
            _VECTOR_RISK.labels("safe").inc(safe)
            if filtered:
                _VECTOR_RISK.labels("filtered").inc(filtered)

            if self.metrics_db is not None:
                try:  # pragma: no cover - legacy path
                    try:
                        self.metrics_db.log_patch_outcome(
                            patch_id or "",
                            result,
                            pairs,
                            session_id=session_id,
                            roi_tag=roi_tag_val.value,
                        )
                    except TypeError:
                        self.metrics_db.log_patch_outcome(
                            patch_id or "", result, pairs, session_id=session_id
                        )
                except Exception:
                    logger.exception("metrics_db.log_patch_outcome failed")
            else:
                if self.patch_db is not None and patch_id:
                    try:
                        self.patch_db.record_provenance(int(patch_id), provenance_meta)
                    except Exception as exc:
                        logger.exception("patch_db.record_provenance failed")
                        errors.append({"db_write": str(exc)})
                    try:
                        self.patch_db.log_ancestry(int(patch_id), detailed_meta)
                    except Exception as exc:
                        logger.exception("patch_db.log_ancestry failed")
                        errors.append({"db_write": str(exc)})
                    try:
                        self.patch_db.log_contributors(
                            int(patch_id), detailed, session_id
                        )
                    except Exception as exc:
                        logger.exception("patch_db.log_contributors failed")
                        errors.append({"db_write": str(exc)})
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
                    logger.exception("vector_metrics.update_outcome failed")
                    raise
                if patch_id:
                    try:
                        self.vector_metrics.record_patch_ancestry(patch_id, vm_vectors)
                    except Exception:
                        logger.exception("vector_metrics.record_patch_ancestry failed")
            for origin, roi in origin_totals.items():
                if self.vector_metrics is not None:
                    try:
                        self.vector_metrics.log_retrieval_feedback(
                            origin, win=result, regret=not result, roi=roi
                        )
                    except Exception:
                        logger.exception("vector_metrics.log_retrieval_feedback failed")
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
                        logger.exception("event bus retrieval feedback publish failed")
                elif UnifiedEventBus is not None:
                    try:
                        UnifiedEventBus().publish("retrieval:feedback", payload)
                    except Exception:
                        logger.exception(
                            "UnifiedEventBus retrieval feedback publish failed"
                        )
            roi_metrics = {}
            roi_deltas = {}
            if origin_totals:
                for origin, roi in origin_totals.items():
                    metrics = {
                        "roi": roi,
                        "win_rate": 1.0 if result else 0.0,
                        "regret_rate": 0.0 if result else 1.0,
                        "alignment_severity": origin_sev.get(origin, 0.0),
                        "semantic_alerts": sorted(origin_alerts.get(origin, [])),
                        "risk_score": origin_similarity.get(origin, 0.0),
                    }
                    if _DB_RISK is not None:
                        try:
                            _DB_RISK.labels(origin_db=origin).set(
                                float(metrics["risk_score"])
                            )
                        except Exception:
                            pass
                    roi_metrics[origin] = metrics
                if self.roi_tracker is not None:
                    for origin, stats in roi_metrics.items():
                        try:
                            # send deltas for each origin individually
                            self.roi_tracker.update_db_metrics({origin: stats})
                        except Exception as exc:
                            logger.exception("ROITracker.update_db_metrics failed")
                            errors.append({"roi_update": str(exc)})
                    try:
                        deltas = self.roi_tracker.origin_db_deltas()
                    except Exception as exc:
                        deltas = {}
                        logger.exception("Failed to fetch ROI tracker origin deltas")
                        errors.append({"roi_update": str(exc)})
                    for origin in origin_totals:
                        val = deltas.get(origin)
                        if val is None:
                            continue
                        try:
                            roi_deltas[origin] = float(val)
                            if _DB_ROI_DELTA is not None:
                                _DB_ROI_DELTA.labels(origin_db=origin).set(float(val))
                        except Exception:
                            if _DB_ROI_DELTA is not None:
                                try:
                                    _DB_ROI_DELTA.labels(origin_db=origin).set(0.0)
                                except Exception:
                                    pass
                else:
                    for origin, stats in roi_metrics.items():
                        payload = {"db": origin, **stats}
                        if self.event_bus is not None:
                            try:
                                self.event_bus.publish("roi:update", payload)
                            except Exception as exc:
                                logger.exception(
                                    "event bus ROI update publish failed",
                                )
                                errors.append({"roi_update": str(exc)})
                        elif UnifiedEventBus is not None:
                            try:
                                UnifiedEventBus().publish("roi:update", payload)
                            except Exception as exc:
                                logger.exception(
                                    "UnifiedEventBus ROI update publish failed",
                                )
                                errors.append({"roi_update": str(exc)})
                    if _DB_ROI_DELTA is not None:
                        for origin, roi in origin_totals.items():
                            try:
                                _DB_ROI_DELTA.labels(origin_db=origin).set(float(roi))
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
                                logger.exception(
                                    "event bus embedding backfill publish failed"
                                )
                    elif UnifiedEventBus is not None:
                        bus = UnifiedEventBus()
                        for origin in unique_origins:
                            try:
                                bus.publish("embedding:backfill", {"db": origin})
                            except Exception:
                                logger.exception(
                                    "UnifiedEventBus embedding backfill publish failed"
                                )
                error_trace_count = len(errors)
                tests_passed_count = 1 if tests_passed else 0
                tests_failed_count = 1 if tests_passed is False else 0
                metrics = EnhancementMetrics(
                    lines_changed=lines_changed or 0,
                    context_tokens=context_tokens_val,
                    time_to_completion=time_to_completion or 0.0,
                    tests_passed=tests_passed_count,
                    tests_failed=tests_failed_count,
                    error_traces=error_trace_count,
                    effort_estimate=effort_estimate_val or 0.0,
                )
                enhancement_score = _compute_enhancement_score(metrics)
                if self.patch_db is not None and patch_id:
                    try:  # pragma: no cover - best effort
                        self.patch_db.record_vector_metrics(
                            session_id,
                            pairs,
                            patch_id=int(patch_id),
                            contribution=0.0 if contribution is None else contribution,
                            roi_delta=roi_delta,
                            win=result,
                            regret=not result,
                            lines_changed=lines_changed,
                            tests_passed=tests_passed,
                            tests_failed_after=tests_failed_count,
                            context_tokens=context_tokens_val,
                            patch_difficulty=patch_difficulty,
                            effort_estimate=effort_estimate_val,
                            enhancement_name=enhancement_name,
                            start_time=start_time,
                            time_to_completion=time_to_completion,
                            timestamp=end_time,
                            errors=errors,
                            error_trace_count=error_trace_count,
                            roi_tag=roi_tag_val.value,
                            diff=diff,
                            summary=summary_arg,
                            outcome=outcome,
                            roi_deltas=roi_deltas,
                            enhancement_score=enhancement_score,
                        )
                    except Exception as exc:
                        logger.exception("patch_db.record_vector_metrics failed")
                        errors.append({"db_write": str(exc)})
                        raise
                # Generate and persist patch embedding for future retrieval
                if self.patch_db is not None and patch_id:
                    try:
                        desc_text = ""
                        embed_diff = diff
                        embed_summary = summary
                        try:
                            rec = self.patch_db.get(int(patch_id))
                            if rec is not None:
                                desc_text = getattr(rec, "description", "") or ""
                                if embed_diff is None:
                                    embed_diff = getattr(rec, "diff", None)
                                if embed_summary is None:
                                    embed_summary = getattr(rec, "summary", None)
                        except Exception:
                            logger.exception("Failed to fetch patch record for embedding")
                        if desc_text or embed_diff or embed_summary:
                            record = {
                                "description": desc_text,
                                "diff": embed_diff or "",
                                "summary": embed_summary or "",
                                "enhancement_score": enhancement_score,
                                "roi_tag": roi_tag_val.value,
                            }
                            svc = self.vector_service
                            if svc is None:
                                try:
                                    from .vectorizer import SharedVectorService  # type: ignore
                                    svc = SharedVectorService(
                                        bootstrap_fast=self.bootstrap_fast
                                    )
                                except Exception:
                                    svc = None
                            if svc is not None:
                                try:
                                    svc.vectorise_and_store(
                                        "patch",
                                        str(patch_id),
                                        record,
                                        origin_db="patch",
                                        metadata=record,
                                    )
                                except Exception:
                                    logger.exception(
                                        "SharedVectorService patch embedding failed",
                                    )
                            else:
                                try:
                                    from .patch_vectorizer import PatchVectorizer  # type: ignore
                                    PatchVectorizer(
                                        bootstrap_fast=self.bootstrap_fast
                                    ).try_add_embedding(int(patch_id), record, "patch")
                                except Exception:
                                    logger.exception(
                                        "PatchVectorizer patch embedding failed",
                                    )
                    except Exception:
                        logger.exception("Failed to embed patch metadata")
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
                            logger.exception("Failed to store lesson metadata")
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
        max_risk = max(origin_similarity.values(), default=0.0)
        payload = {
            "result": result,
            "roi_metrics": roi_metrics,
            "win": result,
            "regret": not result,
            "alignment_severity": max_sev,
            "semantic_alerts": all_alerts,
            "risk_score": max_risk,
            "risk_scores": dict(origin_similarity),
            "enhancement_score": enhancement_score,
            "roi_tag": roi_tag_val.value,
            "start_time": start_time,
            "end_time": end_time,
            "time_to_completion": time_to_completion,
            "roi_delta": roi_delta,
        }
        if patch_id:
            payload["patch_id"] = patch_id
        if risk_flags:
            payload["risk_flags"] = list(risk_flags)
        if diff_risk_score is not None:
            payload["diff_risk_score"] = float(diff_risk_score)

        # Persist risk summaries for audit
        try:
            conn = router.get_connection("risk_summaries")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS risk_summaries("
                "ts REAL, patch_id TEXT, origin TEXT, risk REAL)"
            )
            ts = time.time()
            for origin, risk in origin_similarity.items():
                conn.execute(
                    "INSERT INTO risk_summaries(ts, patch_id, origin, risk) VALUES(?,?,?,?)",
                    (ts, str(patch_id), origin, float(risk)),
                )
            conn.commit()
        except Exception:
            logger.exception("Failed to persist risk summaries")

        if self.event_bus is not None:
            try:
                self.event_bus.publish("patch_logger:outcome", payload)
            except Exception:
                logger.exception(
                    "event bus patch_logger outcome publish failed"
                )
        elif UnifiedEventBus is not None:
            try:
                UnifiedEventBus().publish("patch_logger:outcome", payload)
            except Exception:
                logger.exception(
                    "UnifiedEventBus patch_logger outcome publish failed"
                )
        summary_payload = {
            "patch_id": patch_id,
            "result": result,
            "roi_deltas": dict(roi_deltas),
            "roi_delta": roi_delta,
            "lines_changed": lines_changed,
            "context_tokens": context_tokens_val,
            "patch_difficulty": patch_difficulty,
            "effort_estimate": effort_estimate_val,
            "tests_passed": tests_passed,
            "enhancement_name": enhancement_name,
            "start_time": start_time,
            "end_time": end_time,
            "time_to_completion": time_to_completion,
            "diff": diff,
            "summary": summary,
            "outcome": outcome,
            "errors": errors,
            "error_trace_count": error_trace_count,
            "roi_tag": roi_tag_val.value,
            "enhancement_score": enhancement_score,
        }
        if risk_flags:
            summary_payload["risk_flags"] = list(risk_flags)
        if diff_risk_score is not None:
            summary_payload["diff_risk_score"] = float(diff_risk_score)
        if error_summary is not None:
            summary_payload["error_summary"] = error_summary
        if self.event_bus is not None:
            try:
                self.event_bus.publish("patch:summary", summary_payload)
            except Exception:
                logger.exception("event bus patch summary publish failed")
        elif UnifiedEventBus is not None:
            try:
                UnifiedEventBus().publish("patch:summary", summary_payload)
            except Exception:
                logger.exception("UnifiedEventBus patch summary publish failed")

        if _ps_log_outcome is not None and patch_id:
            try:
                _ps_log_outcome(
                    {
                        "patch_id": patch_id,
                        "result": "ok" if result else "failed",
                        "vectors": [(o or "", vid, s) for o, vid, s in detailed],
                        "retrieval_session_id": session_id,
                        "roi_tag": roi_tag_val.value,
                    }
                )
            except Exception:
                logger.exception("patch score outcome log failed")

        if self.vector_metrics is not None and patch_id:
            try:  # pragma: no cover - best effort
                self.vector_metrics.record_patch_summary(
                    str(patch_id),
                    errors=errors,
                    tests_passed=tests_passed,
                    lines_changed=lines_changed,
                    context_tokens=context_tokens_val,
                    patch_difficulty=patch_difficulty,
                    start_time=start_time,
                    time_to_completion=time_to_completion,
                    error_trace_count=error_trace_count,
                    roi_tag=roi_tag_val.value,
                    effort_estimate=effort_estimate_val,
                    enhancement_score=enhancement_score,
                )
            except Exception:
                logger.exception("vector_metrics.record_patch_summary failed")

        if risk_callback is not None:
            try:
                risk_callback(origin_similarity)
            except Exception:
                logger.exception("risk callback failed")

        _TRACK_FAILURES.inc(error_trace_count)
        if tests_passed is not None:
            _TRACK_TESTS.labels("passed" if tests_passed else "failed").inc()

        if self.weight_adjuster is not None:
            try:
                vector_details = [
                    (o, vid, enhancement_score, roi_tag_val.value)
                    for o, vid, _ in detailed
                ]
                self.weight_adjuster.adjust(
                    vector_details,
                    error_trace_count=error_trace_count,
                    tests_passed=tests_passed,
                )
            except Exception:
                logger.exception("Failed to adjust ranking weights")

        return TrackResult(
            origin_similarity,
            errors=errors,
            tests_passed=tests_passed,
            lines_changed=lines_changed,
            context_tokens=context_tokens_val,
            patch_difficulty=patch_difficulty,
            duration_s=time_to_completion,
            error_count=error_trace_count,
            roi_deltas=roi_deltas,
            roi_delta=roi_delta,
            effort_estimate=effort_estimate_val,
            enhancement_score=enhancement_score,
        )

    # ------------------------------------------------------------------
    def recompute_enhancement_score(
        self,
        patch_id: str | int,
        metrics: EnhancementMetrics,
        *,
        roi_tag: RoiTag | str | None = None,
        errors: Sequence[Mapping[str, Any]] | None = None,
        start_time: float | None = None,
        time_to_completion: float | None = None,
        error_trace_count: int | None = None,
    ) -> float:
        """Recalculate and persist enhancement score for ``patch_id``.

        ``metrics`` should contain the latest patch statistics.  The updated
        score along with the provided ``roi_tag`` and other optional metrics is
        written to both ``PatchHistoryDB`` and ``VectorMetricsDB`` when those
        databases are available.  The new enhancement score is returned.
        """

        score = _compute_enhancement_score(metrics)
        roi_val = RoiTag.validate(roi_tag)

        patch_difficulty = metrics.lines_changed + metrics.context_tokens
        tests_passed = bool(metrics.tests_passed)
        err_count = (
            error_trace_count if error_trace_count is not None else metrics.error_traces
        )
        tt_completion = (
            time_to_completion if time_to_completion is not None else metrics.time_to_completion
        )

        if self.patch_db is not None:
            try:  # pragma: no cover - best effort
                conn = self.patch_db.router.get_connection("patch_history")
                conn.execute(
                    "UPDATE patch_history SET lines_changed=?, tests_passed=?, "
                    "context_tokens=?, patch_difficulty=?, start_time=COALESCE(?, start_time), "
                    "time_to_completion=?, error_trace_count=?, roi_tag=?, effort_estimate=?, "
                    "enhancement_score=? WHERE id=?",
                    (
                        metrics.lines_changed,
                        int(tests_passed),
                        metrics.context_tokens,
                        patch_difficulty,
                        start_time,
                        tt_completion,
                        err_count,
                        roi_val.value,
                        metrics.effort_estimate,
                        score,
                        int(patch_id),
                    ),
                )
                conn.commit()
            except Exception:
                logger.exception(
                    "Failed to update patch history with recomputed score"
                )

        if self.vector_metrics is not None:
            try:  # pragma: no cover - best effort
                self.vector_metrics.record_patch_summary(
                    str(patch_id),
                    errors=errors,
                    tests_passed=tests_passed,
                    lines_changed=metrics.lines_changed,
                    context_tokens=metrics.context_tokens,
                    patch_difficulty=patch_difficulty,
                    start_time=start_time,
                    time_to_completion=tt_completion,
                    error_trace_count=err_count,
                    roi_tag=roi_val.value,
                    effort_estimate=metrics.effort_estimate,
                    enhancement_score=score,
                )
            except Exception:
                logger.exception("vector_metrics.record_patch_summary failed")

        return score

    # ------------------------------------------------------------------
    def get_patch_summary(self, patch_id: str | int) -> Mapping[str, Any] | None:
        """Return stored patch metadata for *patch_id* if available."""
        if self.patch_db is None:
            return None
        try:
            conn = self.patch_db.router.get_connection("patch_history")
            row = conn.execute(
                "SELECT diff, summary, outcome, lines_changed, tests_passed, "
                "context_tokens, patch_difficulty, effort_estimate, enhancement_name, "
                "start_time, time_to_completion, timestamp, roi_deltas, errors, "
                "error_trace_count, roi_tag, enhancement_score FROM patch_history WHERE id=?",
                (int(patch_id),),
            ).fetchone()
            if row is None:
                return None
            (
                diff,
                summary,
                outcome,
                lines_changed,
                tests_passed,
                context_tokens,
                patch_difficulty,
                effort_estimate,
                enhancement_name,
                start_ts,
                duration,
                ts,
                roi_json,
                err_json,
                err_count,
                roi_tag,
                enh_score,
            ) = row
            try:
                roi_data = json.loads(roi_json) if roi_json else {}
            except Exception:
                roi_data = {}
            try:
                err_data = json.loads(err_json) if err_json else []
            except Exception:
                err_data = []
            return {
                "patch_id": int(patch_id),
                "diff": diff,
                "summary": summary,
                "outcome": outcome,
                "lines_changed": lines_changed,
                "tests_passed": bool(tests_passed) if tests_passed is not None else None,
                "context_tokens": context_tokens,
                "patch_difficulty": patch_difficulty,
                "enhancement_name": enhancement_name,
                "effort_estimate": effort_estimate,
                "start_time": start_ts,
                "time_to_completion": duration,
                "timestamp": ts,
                "roi_deltas": roi_data,
                "errors": err_data,
                "error_trace_count": err_count,
                "roi_tag": roi_tag,
                "enhancement_score": enh_score,
            }
        except Exception:
            logger.exception("Failed to retrieve patch summary")
            return None

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
        roi_delta: float | None = None,
        retrieval_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        risk_callback: "Callable[[Mapping[str, float]], Any]" | None = None,
        lines_changed: int | None = None,
        tests_passed: bool | None = None,
        enhancement_name: str | None = None,
        context_tokens: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        diff: str | None = None,
        summary: str | None = None,
        outcome: str | None = None,
        error_summary: str | None = None,
        error_traces: Sequence[Mapping[str, Any]] | None = None,
        roi_tag: RoiTag | str | None = None,
        effort_estimate: float | None = None,
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
            roi_delta=roi_delta,
            retrieval_metadata=retrieval_metadata,
            risk_callback=risk_callback,
            lines_changed=lines_changed,
            tests_passed=tests_passed,
            enhancement_name=enhancement_name,
            context_tokens=context_tokens,
            start_time=start_time,
            end_time=end_time,
            diff=diff,
            summary=summary,
            outcome=outcome,
            error_summary=error_summary,
            error_traces=error_traces,
            roi_tag=roi_tag,
            effort_estimate=effort_estimate,
        )


__all__ = ["PatchLogger", "RoiTag"]
