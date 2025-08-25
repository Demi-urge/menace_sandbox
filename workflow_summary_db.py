from __future__ import annotations

"""SQLite-backed store for workflow summaries.

This module persists short textual summaries for workflows in a shared
``workflow_summaries`` table. Records are tagged with the originating
``menace_id`` so multiple Menace instances can safely share the same
storage. Read operations respect a ``scope`` parameter (``"local"`` by
default) to control which menace IDs are visible:

- ``"local"`` – only records created by the current menace
- ``"global"`` – records from other Menace instances
- ``"all"`` – no menace ID filtering

Examples::

    db = WorkflowSummaryDB()
    db.get_summary(1, scope="local")   # current menace only
    db.get_summary(1, scope="global")  # summaries from other menaces
    db.all_summaries(scope="all")      # no menace filtering

The ``scope`` parameter replaces the deprecated ``include_cross_instance`` and
``all_instances`` flags.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Literal

try:  # pragma: no cover - import available in package context
    from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
    from .scope_utils import Scope, build_scope_clause, apply_scope
except Exception:  # pragma: no cover - fallback for tests
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
    from scope_utils import Scope, build_scope_clause, apply_scope

MENACE_ID = "workflow_summary_db"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)


@dataclass
class WorkflowSummary:
    """Simple data container for a workflow summary."""

    workflow_id: int
    summary: str
    source_menace_id: str = ""
    timestamp: str | None = None


class WorkflowSummaryDB:
    """Persist workflow summaries routed via :class:`DBRouter`."""

    def __init__(self, *, router: DBRouter | None = None) -> None:
        self.router = router or DB_ROUTER
        self.conn = self.router.get_connection("workflow_summaries")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_summaries(
                workflow_id INTEGER PRIMARY KEY,
                summary TEXT,
                timestamp TEXT,
                source_menace_id TEXT NOT NULL
            )
            """,
        )
        # Lightweight migration: add timestamp column if missing
        cur = self.conn.execute("PRAGMA table_info(workflow_summaries)")
        columns = [row[1] for row in cur.fetchall()]
        if "timestamp" not in columns:
            self.conn.execute(
                "ALTER TABLE workflow_summaries ADD COLUMN timestamp TEXT"
            )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflow_summaries_source_menace_id
                ON workflow_summaries(source_menace_id)
            """,
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflow_summaries_timestamp
                ON workflow_summaries(timestamp)
            """,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def _current_menace_id(self, source_menace_id: str | None = None) -> str:
        return source_menace_id or (self.router.menace_id if self.router else "")

    # ------------------------------------------------------------------
    def set_summary(
        self,
        workflow_id: int,
        summary: str,
        *,
        source_menace_id: str | None = None,
    ) -> None:
        """Insert or update the summary for ``workflow_id``."""

        menace_id = self._current_menace_id(source_menace_id)
        conn = self.router.get_connection("workflow_summaries")
        ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO workflow_summaries(
                source_menace_id, workflow_id, summary, timestamp
            )
            VALUES(?,?,?,?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                summary=excluded.summary,
                timestamp=excluded.timestamp
            """,
            (menace_id, workflow_id, summary, ts),
        )
        conn.commit()

    # ------------------------------------------------------------------
    def get_summary(
        self,
        workflow_id: int,
        *,
        scope: Literal["local", "global", "all"] = "local",
    ) -> WorkflowSummary | None:
        """Return the summary for ``workflow_id`` filtered by ``scope``."""

        menace_id = self._current_menace_id()
        clause, params = build_scope_clause(
            "workflow_summaries", Scope(scope), menace_id
        )
        conn = self.router.get_connection("workflow_summaries")
        query = apply_scope(
            "SELECT workflow_id, summary, source_menace_id, timestamp"
            " FROM workflow_summaries WHERE workflow_id=?",
            clause,
        )
        params = [workflow_id] + params
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return WorkflowSummary(**dict(row)) if row else None

    # ------------------------------------------------------------------
    def all_summaries(
        self, *, scope: Literal["local", "global", "all"] = "local"
    ) -> list[WorkflowSummary]:
        """Return all summaries visible under the selected ``scope``."""

        menace_id = self._current_menace_id()
        clause, params = build_scope_clause(
            "workflow_summaries", Scope(scope), menace_id
        )
        conn = self.router.get_connection("workflow_summaries")
        query = apply_scope(
            "SELECT workflow_id, summary, source_menace_id, timestamp"
            " FROM workflow_summaries",
            clause,
        )
        cur = conn.execute(query, params)
        return [WorkflowSummary(**dict(row)) for row in cur.fetchall()]
