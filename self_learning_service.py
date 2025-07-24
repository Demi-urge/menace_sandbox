from __future__ import annotations

"""Background service for automatic incremental learning."""

import time
import logging
from typing import Optional
from threading import Event, Thread

from .unified_event_bus import UnifiedEventBus
from .neuroplasticity import PathwayDB
from .menace_memory_manager import MenaceMemoryManager
from .code_database import CodeDB
from .resource_allocation_optimizer import ROIDB
from .learning_engine import LearningEngine
from .unified_learning_engine import UnifiedLearningEngine
from .action_learning_engine import ActionLearningEngine
from .self_learning_coordinator import SelfLearningCoordinator

logger = logging.getLogger(__name__)


def main(
    persist_events: Optional[str] = None,
    *,
    stop_event: Event | None = None,
    persist_progress: Optional[str] = None,
) -> None:
    """Run the self-learning coordinator until ``stop_event`` is set."""

    bus = UnifiedEventBus(persist_events) if persist_events else UnifiedEventBus()

    pdb = PathwayDB(event_bus=bus)
    mm = MenaceMemoryManager(event_bus=bus)
    code_db = CodeDB(event_bus=bus)
    roi_db = ROIDB()

    le = LearningEngine(pdb, mm)
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi_db)
    ale = ActionLearningEngine(pdb, roi_db, code_db, ule)

    coord = SelfLearningCoordinator(
        bus, learning_engine=le, unified_engine=ule, action_engine=ale
    )
    coord.start()

    try:
        while stop_event is None or not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        if stop_event is not None:
            stop_event.set()
    finally:
        coord.stop()
        if persist_progress:
            try:
                results = coord.evaluation_manager.evaluate_all()
                import json

                with open(persist_progress, "w", encoding="utf-8") as fh:
                    json.dump(results, fh)
            except Exception as exc:  # pragma: no cover - persistence failures
                logger.exception("failed to persist progress: %s", exc)


__all__ = ["main"]


def run_background(
    persist_events: Optional[str] = None,
    *,
    persist_progress: Optional[str] = None,
) -> tuple[callable, callable]:
    """Return ``start`` and ``stop`` callables to manage the background thread."""

    stop_event = Event()
    thread: Thread | None = None

    def _target() -> None:
        main(
            persist_events,
            stop_event=stop_event,
            persist_progress=persist_progress,
        )

    def start() -> None:
        nonlocal thread
        if thread is None:
            thread = Thread(target=_target, daemon=True)
            thread.start()

    def stop() -> None:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=0)

    return start, stop


__all__.append("run_background")
