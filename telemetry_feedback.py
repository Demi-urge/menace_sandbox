from __future__ import annotations

"""Automatic patching feedback loop using telemetry."""

from pathlib import Path
import sqlite3
import threading
import time
from typing import Optional, Dict, List, Tuple

from .error_logger import ErrorLogger
from .self_coding_engine import SelfCodingEngine


class TelemetryFeedback:
    """Monitor telemetry and trigger self-coding patches."""

    def __init__(
        self,
        logger: ErrorLogger,
        engine: SelfCodingEngine,
        *,
        threshold: int = 3,
        interval: int = 60,
    ) -> None:
        self.logger = logger
        self.engine = engine
        self.threshold = threshold
        self.interval = interval
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
        rows = self._recent_events()
        grouped: Dict[str, List[Tuple[int, str, str, str]]] = {}
        for row in rows:
            grouped.setdefault(row[1] or "", []).append(row)
        if not grouped:
            return
        error_type, events = max(grouped.items(), key=lambda kv: len(kv[1]))
        if len(events) < self.threshold:
            return
        root_module = events[0][3] or self.engine.bot_name
        path = Path(f"{root_module}.py")
        if not path.exists():
            return
        stack = events[0][2] or ""
        line = stack.strip().splitlines()
        desc_tail = line[-1] if line else error_type
        desc = f"fix {error_type}: {desc_tail[:50]}"
        try:
            patch_id, _, _ = self.engine.apply_patch(path, desc)
        except Exception:
            patch_id = None
        self._mark_attempt(events, patch_id)

    # --------------------------------------------------------------
    def _recent_events(self) -> List[Tuple[int, str, str, str]]:
        with sqlite3.connect(self.logger.db.path) as conn:
            rows = conn.execute(
                "SELECT id, error_type, stack_trace, root_module "
                "FROM telemetry WHERE resolution_status='unresolved' "
                "ORDER BY id DESC LIMIT ?",
                (self.threshold * 5,),
            ).fetchall()
        return [(int(r[0]), str(r[1] or ""), str(r[2] or ""), str(r[3] or "")) for r in rows]

    def _mark_attempt(self, events: List[Tuple[int, str, str, str]], patch_id: int | None) -> None:
        with sqlite3.connect(self.logger.db.path) as conn:
            for ev in events:
                conn.execute(
                    "UPDATE telemetry SET patch_id=?, resolution_status='attempted' WHERE id=?",
                    (patch_id, ev[0]),
                )
            conn.commit()


__all__ = ["TelemetryFeedback"]
