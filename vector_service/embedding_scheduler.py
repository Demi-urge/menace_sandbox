from __future__ import annotations

"""Periodic scheduler for :class:`EmbeddingBackfill`."""

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Optional, Dict, List

from unified_event_bus import UnifiedEventBus

from patch_safety import PatchSafety
from .embedding_backfill import EmbeddingBackfill, KNOWN_DB_KINDS

try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running standalone
    import metrics_exporter as _me  # type: ignore

_SCHED_OUTCOME = _me.Gauge(
    "embedding_scheduler_runs_total",
    "Outcomes of scheduled EmbeddingBackfill runs",
    labelnames=["status"],
)
_SCHED_DURATION = _me.Gauge(
    "embedding_scheduler_run_duration_seconds",
    "Duration of scheduled EmbeddingBackfill runs",
)

# Database kinds recognised for on-demand backfill events.
_SUPPORTED_SOURCES = set(KNOWN_DB_KINDS)


class EmbeddingScheduler:
    """Invoke :meth:`EmbeddingBackfill.run` on a timer."""

    def __init__(
        self,
        *,
        interval: float = 86400,
        batch_size: Optional[int] = None,
        backend: Optional[str] = None,
        sources: Optional[list[str]] = None,
        stale_threshold: float = 0.0,
        event_bus: Optional[UnifiedEventBus] = None,
        event_topic: str = "db:new_record",
        event_batch: int = 1,
        event_throttle: float = 0.0,
        patch_safety: PatchSafety | None = None,
    ) -> None:
        self.interval = interval
        self.batch_size = batch_size
        self.backend = backend
        self.sources = sources or ["code", "bot", "error", "workflow"]
        self.stale_threshold = stale_threshold
        self.event_bus = event_bus or UnifiedEventBus()
        self.event_topic = event_topic
        self.event_batch = max(1, event_batch)
        self.event_throttle = max(0.0, event_throttle)
        self._event_counts: defaultdict[str, int] = defaultdict(int)
        self._last_event_run: defaultdict[str, float] = defaultdict(float)
        self._thread: Optional[threading.Thread] = None
        self.running = False
        self.backfill = EmbeddingBackfill()
        self.patch_safety = patch_safety or PatchSafety()

        try:  # subscribe to on-demand backfill events
            self.event_bus.subscribe(self.event_topic, self._handle_event)
        except Exception:  # pragma: no cover - best effort
            logging.getLogger(__name__).exception("failed subscribing to event bus")

    # ------------------------------------------------------------------
    def _handle_event(self, _topic: str, payload: object) -> None:
        """Handle on-demand backfill requests from the event bus."""
        source = None
        meta = None
        if isinstance(payload, dict):
            source = payload.get("source")
            meta = payload.get("metadata") or payload.get("meta")
        else:
            source = getattr(payload, "source", None)
            meta = getattr(payload, "metadata", getattr(payload, "meta", None))
        if not source or source not in _SUPPORTED_SOURCES:
            return
        if meta is not None and not self.patch_safety.pre_embed_check(meta):
            return
        self._event_counts[source] += 1
        now = time.time()
        last = self._last_event_run[source]
        if self._event_counts[source] < self.event_batch:
            return
        if self.event_throttle and now - last < self.event_throttle:
            return
        self._event_counts[source] = 0
        self._last_event_run[source] = now
        try:
            EmbeddingBackfill().run(dbs=[source])
            try:
                self.patch_safety.load_failures(force=True)
            except Exception:  # pragma: no cover - best effort
                logging.getLogger(__name__).exception(
                    "patch safety refresh failed"
                )
        except Exception:  # pragma: no cover - best effort
            logging.getLogger(__name__).exception("on-demand embedding backfill failed")

    def _collect_status(self, db_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Gather counts and staleness metrics for ``db_names``."""

        report: Dict[str, Dict[str, float]] = {}
        subclasses = self.backfill._load_known_dbs(names=db_names)
        for cls in subclasses:
            try:
                db = cls(vector_backend=self.backend) if self.backend else cls()  # type: ignore[call-arg]
            except Exception:
                try:
                    db = cls()  # type: ignore[call-arg]
                except Exception:
                    continue
            records = 0
            stale = 0
            for rid, record, _ in db.iter_records():
                records += 1
                if db.needs_refresh(rid, record):
                    stale += 1
            vectors = len(getattr(db, "_id_map", []))
            kind = cls.__name__.lower()
            if kind.endswith("db"):
                kind = kind[:-2]
            ratio = stale / records if records else 0.0
            report[kind] = {
                "records": records,
                "vectors": vectors,
                "stale": stale,
                "stale_ratio": ratio,
            }
        return report

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        logger = logging.getLogger(__name__)
        while self.running:
            start = time.time()
            status = "success"
            try:
                db_names = self.sources or []
                report = self._collect_status(db_names)
                logger.info(
                    "embedding scheduler report",
                    extra={"embedding_report": report},
                )
                to_backfill = [
                    name
                    for name, stats in report.items()
                    if stats["records"] != stats["vectors"]
                    or stats["stale_ratio"] > self.stale_threshold
                ]
                if to_backfill:
                    logger.info(
                        "embedding scheduler triggering backfill",
                        extra={"backend": self.backend, "dbs": to_backfill},
                    )
                    self.backfill.run(
                        batch_size=self.batch_size,
                        backend=self.backend,
                        dbs=to_backfill,
                    )
                    try:
                        self.patch_safety.load_failures(force=True)
                    except Exception:  # pragma: no cover - best effort
                        logging.getLogger(__name__).exception(
                            "patch safety refresh failed",
                        )
            except Exception:  # pragma: no cover - best effort
                status = "failure"
                logging.exception("embedding backfill run failed")
            _SCHED_OUTCOME.labels(status).inc()
            _SCHED_DURATION.set(time.time() - start)
            end = time.time() + self.interval
            while self.running and time.time() < end:
                time.sleep(1)
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=0)
            self._thread = None


# ---------------------------------------------------------------------------

def start_scheduler_from_env() -> EmbeddingScheduler | None:
    """Start :class:`EmbeddingScheduler` based on environment variables."""

    interval = float(os.getenv("EMBEDDING_SCHEDULER_INTERVAL", "86400"))
    if interval <= 0:
        return None
    batch = os.getenv("EMBEDDING_SCHEDULER_BATCH_SIZE")
    backend = os.getenv("EMBEDDING_SCHEDULER_BACKEND")
    sources_env = os.getenv("EMBEDDING_SCHEDULER_SOURCES", "")
    sources = [s.strip() for s in sources_env.split(",") if s.strip()]
    if not sources:
        sources = None
    event_batch = os.getenv("EMBEDDING_SCHEDULER_EVENT_BATCH")
    event_throttle = os.getenv("EMBEDDING_SCHEDULER_EVENT_THROTTLE")
    stale_threshold = os.getenv("EMBEDDING_SCHEDULER_STALE_THRESHOLD")
    scheduler = EmbeddingScheduler(
        interval=interval,
        batch_size=int(batch) if batch else None,
        backend=backend,
        sources=sources,
        stale_threshold=float(stale_threshold) if stale_threshold else 0.0,
        event_batch=int(event_batch) if event_batch else 1,
        event_throttle=float(event_throttle) if event_throttle else 0.0,
    )
    scheduler.start()
    return scheduler


def run_backfill_from_env() -> None:
    """Run a one-shot :class:`EmbeddingBackfill` based on environment variables."""

    batch = os.getenv("EMBEDDING_SCHEDULER_BATCH_SIZE")
    backend = os.getenv("EMBEDDING_SCHEDULER_BACKEND")
    sources_env = os.getenv("EMBEDDING_SCHEDULER_SOURCES", "")
    sources = [s.strip() for s in sources_env.split(",") if s.strip()] or None
    EmbeddingBackfill().run(
        batch_size=int(batch) if batch else None,
        backend=backend,
        dbs=sources,
    )


__all__ = ["EmbeddingScheduler", "start_scheduler_from_env", "run_backfill_from_env"]


if __name__ == "__main__":  # pragma: no cover - convenience for cron
    scheduler = start_scheduler_from_env()
    if scheduler is None:
        run_backfill_from_env()
    else:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            scheduler.stop()
