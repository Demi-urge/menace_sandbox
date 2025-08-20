from __future__ import annotations

"""Periodic scheduler for :class:`EmbeddingBackfill`."""

import logging
import os
import threading
import time
from typing import Sequence, Optional

from .embedding_backfill import EmbeddingBackfill

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


class EmbeddingScheduler:
    """Invoke :meth:`EmbeddingBackfill.run` on a timer."""

    def __init__(
        self,
        *,
        interval: float = 86400,
        batch_size: Optional[int] = None,
        backend: Optional[str] = None,
        dbs: Sequence[str] | None = None,
    ) -> None:
        self.interval = interval
        self.batch_size = batch_size
        self.backend = backend
        self.dbs = list(dbs) if dbs else None
        self._thread: Optional[threading.Thread] = None
        self.running = False
        self.backfill = EmbeddingBackfill()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        logger = logging.getLogger(__name__)
        while self.running:
            start = time.time()
            status = "success"
            try:
                logger.info(
                    "embedding scheduler triggering backfill",
                    extra={"dbs": self.dbs, "backend": self.backend},
                )
                self.backfill.run(
                    batch_size=self.batch_size,
                    backend=self.backend,
                    dbs=self.dbs,
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
    dbs_env = os.getenv("EMBEDDING_SCHEDULER_DBS")
    dbs = [d.strip() for d in dbs_env.split(",") if d.strip()] if dbs_env else None
    scheduler = EmbeddingScheduler(
        interval=interval,
        batch_size=int(batch) if batch else None,
        backend=backend,
        dbs=dbs,
    )
    scheduler.start()
    return scheduler


__all__ = ["EmbeddingScheduler", "start_scheduler_from_env"]
