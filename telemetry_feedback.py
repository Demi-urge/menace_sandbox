from __future__ import annotations

"""Automatic patching feedback loop using telemetry."""

from pathlib import Path
import sqlite3
import threading
import time
from typing import Optional, List, Tuple

from .error_logger import ErrorLogger
from .self_coding_engine import SelfCodingEngine
from .knowledge_graph import KnowledgeGraph


class TelemetryFeedback:
    """Monitor telemetry and trigger self-coding patches."""

    def __init__(
        self,
        logger: ErrorLogger,
        engine: SelfCodingEngine,
        *,
        threshold: int = 3,
        interval: int = 60,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.logger = logger
        self.engine = engine
        self.threshold = threshold
        self.interval = interval
        self.graph = graph
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)

    # --------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            self.check()
            time.sleep(self.interval)

    def check(self) -> None:
        info = self.logger.db.top_error_module(unresolved_only=True)
        if not info:
            return
        error_type, module, mods, count, sample_bot = info
        if count < self.threshold:
            return
        path = Path(f"{module}.py")
        if not path.exists():
            return
        desc = f"fix {error_type}: {module}"
        try:
            patch_id, _, _ = self.engine.apply_patch(path, desc)
        except Exception:
            patch_id = None
        ids = [
            int(r[0])
            for r in self.logger.db.conn.execute(
                "SELECT id FROM telemetry WHERE error_type=? AND module=? AND resolution_status='unresolved'",
                (error_type, module),
            ).fetchall()
        ]
        events = [(i, error_type, "", module) for i in ids]
        self._mark_attempt(events, patch_id)
        if self.graph:
            try:
                self.graph.add_telemetry_event(sample_bot, error_type, module, mods, patch_id=patch_id)
                self.graph.update_error_stats(self.logger.db)
            except Exception:
                pass

    def _mark_attempt(self, events: List[Tuple[int, str, str, str]], patch_id: int | None) -> None:
        with sqlite3.connect(self.logger.db.path) as conn:
            for ev in events:
                conn.execute(
                    "UPDATE telemetry SET patch_id=?, resolution_status='attempted' WHERE id=?",
                    (patch_id, ev[0]),
                )
            conn.commit()


__all__ = ["TelemetryFeedback"]
