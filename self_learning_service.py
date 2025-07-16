from __future__ import annotations

"""Background service for automatic incremental learning."""

import time
from typing import Optional
from threading import Event

from .unified_event_bus import UnifiedEventBus
from .neuroplasticity import PathwayDB
from .menace_memory_manager import MenaceMemoryManager
from .code_database import CodeDB
from .resource_allocation_optimizer import ROIDB
from .learning_engine import LearningEngine
from .unified_learning_engine import UnifiedLearningEngine
from .action_learning_engine import ActionLearningEngine
from .self_learning_coordinator import SelfLearningCoordinator


def main(persist_events: Optional[str] = None, *, stop_event: Event | None = None) -> None:
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


__all__ = ["main"]
