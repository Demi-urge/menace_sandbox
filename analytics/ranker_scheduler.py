from __future__ import annotations

"""Periodic scheduler for retraining the vector ranker.

The scheduler invokes :func:`analytics.retrain_vector_ranker.retrain_and_reload`
on a timer and listens to ``UnifiedEventBus`` for ``retrieval:feedback``
notifications so that heavy ROI changes can trigger an immediate retrain.
"""

from importlib import import_module
from pathlib import Path
import os
import threading
import time
from typing import Any, Sequence

# Topic emitted by ``UnifiedEventBus`` when ROI feedback is recorded.
FEEDBACK_TOPIC = "retrieval:feedback"

try:  # pragma: no cover - optional dependency
    from unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback when executed directly
    UnifiedEventBus = None  # type: ignore

from . import retrain_vector_ranker as rvr


class RankerScheduler:
    """Periodically retrain the vector ranker model."""

    def __init__(
        self,
        services: Sequence[Any],
        *,
        vector_db: Path | str = "vector_metrics.db",
        patch_db: Path | str = "roi.db",
        model_dir: Path | str = rvr.MODEL_PATH.parent,
        interval: float = 86400,
        roi_threshold: float | None = None,
        event_bus: "UnifiedEventBus" | None = None,
    ) -> None:
        self.services = list(services)
        self.vector_db = Path(vector_db)
        self.patch_db = Path(patch_db)
        self.model_dir = Path(model_dir)
        self.interval = max(0.0, float(interval))
        self.roi_threshold = roi_threshold
        self.event_bus = event_bus
        self.running = False
        self._thread: threading.Thread | None = None
        self._roi_total = 0.0

        if self.event_bus is None and UnifiedEventBus is not None:
            try:
                self.event_bus = UnifiedEventBus()
            except Exception:
                self.event_bus = None
        if self.event_bus is not None:
            try:
                self.event_bus.subscribe(FEEDBACK_TOPIC, self._handle_feedback)
            except Exception:
                pass

    def _handle_feedback(self, _topic: str, event: object) -> None:
        """Trigger retrain when cumulative ROI exceeds ``roi_threshold``."""
        if self.roi_threshold is None:
            return
        roi = 0.0
        if isinstance(event, dict):
            try:
                roi = float(event.get("roi", 0.0))
            except Exception:
                roi = 0.0
        else:
            roi = float(getattr(event, "roi", 0.0) or 0.0)
        self._roi_total += roi
        if abs(self._roi_total) >= self.roi_threshold:
            self.retrain()
            self._roi_total = 0.0

    def retrain(self) -> None:
        """Invoke :func:`retrain_vector_ranker.retrain_and_reload`."""
        rvr.retrain_and_reload(
            self.services,
            vector_db=self.vector_db,
            patch_db=self.patch_db,
            model_dir=self.model_dir,
        )

    def _loop(self) -> None:
        while self.running:
            self.retrain()
            if self.interval <= 0:
                break
            end = time.time() + self.interval
            while self.running and time.time() < end:
                time.sleep(1)

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


def _import_obj(path: str) -> Any:
    mod_name, attr = path.rsplit(":", 1)
    mod = import_module(mod_name)
    return getattr(mod, attr)


def start_scheduler_from_env(services: Sequence[Any] | None = None) -> RankerScheduler | None:
    """Start :class:`RankerScheduler` based on environment variables."""
    interval = float(os.getenv("RANKER_SCHEDULER_INTERVAL", "0"))
    if interval <= 0:
        return None
    vector_db = os.getenv("RANKER_SCHEDULER_VECTOR_DB", "vector_metrics.db")
    patch_db = os.getenv("RANKER_SCHEDULER_PATCH_DB", "roi.db")
    model_dir = os.getenv("RANKER_SCHEDULER_MODEL_DIR", str(rvr.MODEL_PATH.parent))
    roi_thresh = os.getenv("RANKER_SCHEDULER_ROI_THRESHOLD")
    service_paths = [s.strip() for s in os.getenv("RANKER_SCHEDULER_SERVICES", "").split(",") if s.strip()]
    svc_instances: list[Any] = list(services or [])
    for path in service_paths:
        try:
            svc_instances.append(_import_obj(path))
        except Exception:
            continue
    sched = RankerScheduler(
        svc_instances,
        vector_db=vector_db,
        patch_db=patch_db,
        model_dir=model_dir,
        interval=interval,
        roi_threshold=float(roi_thresh) if roi_thresh else None,
    )
    sched.start()
    return sched


__all__ = ["RankerScheduler", "start_scheduler_from_env"]


if __name__ == "__main__":  # pragma: no cover - convenience for cron
    scheduler = start_scheduler_from_env()
    if scheduler is not None:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            scheduler.stop()
