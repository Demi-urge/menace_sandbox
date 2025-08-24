from __future__ import annotations

"""Automatic patching feedback loop using telemetry."""

from pathlib import Path
import threading
import time
from typing import Optional, List, Tuple

from .db_router import (
    DBRouter,
    GLOBAL_ROUTER,
    SHARED_TABLES,
    init_db_router,
)
from .scope_utils import Scope, build_scope_clause, apply_scope

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
        router: DBRouter | None = None,
    ) -> None:
        self.logger = logger
        self.engine = engine
        self.threshold = threshold
        self.interval = interval
        self.graph = graph
        self.router = (
            router
            or getattr(logger.db, "router", None)
            or GLOBAL_ROUTER
            or init_db_router("telemetry_fb")
        )
        SHARED_TABLES.add("telemetry")
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
            self._run_cycle()
            time.sleep(self.interval)

    def check(
        self,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> None:
        self._run_cycle(scope=scope, source_menace_id=source_menace_id)

    def _run_cycle(
        self,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> None:
        info = self.logger.db.top_error_module(
            unresolved_only=True, scope=scope, source_menace_id=source_menace_id
        )
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
            patch_id, _, _ = self.engine.apply_patch(
                path,
                desc,
                reason=desc,
                trigger="telemetry_feedback",
            )
        except Exception:
            patch_id = None
        menace_id = self.logger.db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        query = apply_scope(
            (
                "SELECT id FROM telemetry "
                "WHERE error_type=? AND module=? "
                "AND resolution_status='unresolved'"
            ),
            clause,
        )
        cur = self.logger.db.conn.execute(query, [error_type, module, *params])
        ids = [int(r[0]) for r in cur.fetchall()]
        events = [(i, error_type, "", module) for i in ids]
        self._mark_attempt(events, patch_id)
        if self.graph:
            try:
                self.graph.add_telemetry_event(
                    sample_bot, error_type, module, mods, patch_id=patch_id
                )
                self.graph.update_error_stats(self.logger.db)
            except Exception:
                pass

    def _mark_attempt(self, events: List[Tuple[int, str, str, str]], patch_id: int | None) -> None:
        conn = self.logger.db.conn
        for ev in events:
            conn.execute(
                "UPDATE telemetry SET patch_id=?, resolution_status='attempted' WHERE id=?",
                (patch_id, ev[0]),
            )
        conn.commit()


__all__ = ["TelemetryFeedback"]
