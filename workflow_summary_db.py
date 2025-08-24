from __future__ import annotations

"""SQLite-backed store for workflow summaries.

This module persists short textual summaries for workflows in a shared
``workflow_summaries`` table.  Records are tagged with the originating
``menace_id`` so multiple Menace instances can safely share the same
storage.  Read operations filter by ``source_menace_id`` by default.
"""

from dataclasses import dataclass
import sqlite3

try:  # pragma: no cover - import available in package context
    from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
except Exception:  # pragma: no cover - fallback for tests
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

MENACE_ID = "workflow_summary_db"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)


@dataclass
class WorkflowSummary:
    """Simple data container for a workflow summary."""

    workflow_id: int
    summary: str
    source_menace_id: str = ""


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
                source_menace_id TEXT NOT NULL
            )
            """,
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflow_summaries_source_menace_id
                ON workflow_summaries(source_menace_id)
            """,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def _current_menace_id(self, source_menace_id: str | None) -> str:
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
        conn.execute(
            """
            INSERT INTO workflow_summaries(workflow_id, summary, source_menace_id)
            VALUES(?,?,?)
            ON CONFLICT(workflow_id) DO UPDATE SET summary=excluded.summary
            """,
            (workflow_id, summary, menace_id),
        )
        conn.commit()

    # ------------------------------------------------------------------
    def get_summary(
        self, workflow_id: int, *, source_menace_id: str | None = None
    ) -> str | None:
        """Return the summary for ``workflow_id`` filtered by menace id."""

        menace_id = self._current_menace_id(source_menace_id)
        conn = self.router.get_connection("workflow_summaries")
        cur = conn.execute(
            """
            SELECT summary FROM workflow_summaries
            WHERE workflow_id=? AND source_menace_id=?
            """,
            (workflow_id, menace_id),
        )
        row = cur.fetchone()
        return row["summary"] if row else None

    # ------------------------------------------------------------------
    def all_summaries(
        self, *, source_menace_id: str | None = None
    ) -> list[WorkflowSummary]:
        """Return all summaries for the current menace instance."""

        menace_id = self._current_menace_id(source_menace_id)
        conn = self.router.get_connection("workflow_summaries")
        cur = conn.execute(
            """
            SELECT workflow_id, summary, source_menace_id
            FROM workflow_summaries
            WHERE source_menace_id=?
            """,
            (menace_id,),
        )
        return [WorkflowSummary(**dict(row)) for row in cur.fetchall()]
