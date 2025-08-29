from __future__ import annotations

"""Background service for automatic incremental learning."""

import os
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
from .local_knowledge_module import init_local_knowledge

logger = logging.getLogger(__name__)

# Default number of interactions added to GPT memory before a compaction is
# triggered.  This value can be overridden at runtime via the ``prune_interval``
# parameter or the ``PRUNE_INTERVAL`` environment variable.
DEFAULT_PRUNE_INTERVAL = 50


def main(
    persist_events: Optional[str] = None,
    *,
    stop_event: Event | None = None,
    persist_progress: Optional[str] = None,
    prune_interval: int | None = None,
) -> None:
    """Run the self-learning coordinator until ``stop_event`` is set.

    Parameters
    ----------
    persist_events:
        Path to persist event bus data, if desired.
    stop_event:
        Event used to signal shutdown of the service.
    persist_progress:
        Optional path for writing evaluation results on shutdown.
    prune_interval:
        Number of new GPT memory interactions that triggers a database
        compaction.  Lower values prune more frequently (using more CPU) while
        higher values prune less often (using more disk).  Defaults to
        ``DEFAULT_PRUNE_INTERVAL`` or the value of the ``PRUNE_INTERVAL``
        environment variable.  Must be a positive integer.
    """

    bus = UnifiedEventBus(persist_events) if persist_events else UnifiedEventBus()

    pdb = PathwayDB(event_bus=bus)
    mm = MenaceMemoryManager(event_bus=bus)
    code_db = CodeDB(event_bus=bus)
    roi_db = ROIDB()

    gpt_mem = init_local_knowledge(
        os.getenv("GPT_MEMORY_DB", "gpt_memory.db")
    ).memory

    le = LearningEngine(pdb, mm)
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi_db)
    ale = ActionLearningEngine(pdb, roi_db, code_db, ule)

    coord = SelfLearningCoordinator(
        bus, learning_engine=le, unified_engine=ule, action_engine=ale
    )
    coord.start()

    # Determine the effective prune interval from argument or environment.
    if prune_interval is None:
        env_value = os.getenv("PRUNE_INTERVAL", str(DEFAULT_PRUNE_INTERVAL))
        try:
            prune_interval = int(env_value)
        except ValueError as exc:  # pragma: no cover - validation
            raise ValueError("prune_interval must be an integer") from exc
    if prune_interval <= 0:
        raise ValueError("prune_interval must be positive")

    def _prune_task() -> None:
        last = 0
        while stop_event is None or not stop_event.is_set():
            try:
                cur = gpt_mem.conn.execute("SELECT COUNT(*) FROM interactions")
                count = cur.fetchone()[0]
            except Exception:
                count = 0
            if count - last >= prune_interval:
                try:
                    gpt_mem.compact(prune_interval)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("scheduled prune failed")
                last = count
            time.sleep(1)

    prune_thread = Thread(target=_prune_task, daemon=True)
    prune_thread.start()

    try:
        while stop_event is None or not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        if stop_event is not None:
            stop_event.set()
    finally:
        coord.stop()
        prune_thread.join(timeout=0)
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
    prune_interval: int | None = None,
) -> tuple[callable, callable]:
    """Return ``start`` and ``stop`` callables to manage the background thread.

    Parameters
    ----------
    persist_events: optional str
        Path to persist event bus data, if desired.
    persist_progress: optional str
        Optional path for writing evaluation results on shutdown.
    prune_interval: optional int
        Passed through to :func:`main` to control GPT memory compaction
        frequency.  See :func:`main` for tuning guidance and validation rules.
    """

    stop_event = Event()
    thread: Thread | None = None

    def _target() -> None:
        main(
            persist_events,
            stop_event=stop_event,
            persist_progress=persist_progress,
            prune_interval=prune_interval,
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
