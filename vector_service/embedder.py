from __future__ import annotations

"""Embed patch history records using :class:`SharedVectorService`."""

from pathlib import Path
from typing import Any

from code_database import PatchHistoryDB
from .vectorizer import SharedVectorService


class Embedder:
    """Iterate patch history and persist text embeddings."""

    def __init__(self, path: str | Path | None = None, svc: SharedVectorService | None = None) -> None:
        self.db = PatchHistoryDB(path)
        self.conn = self.db.router.get_connection("patch_history")
        self.svc = svc or SharedVectorService()

    @staticmethod
    def _compose(record: Any) -> str:
        """Return text from patch ``record`` by joining key fields."""
        if isinstance(record, dict):
            desc = record.get("description") or ""
            diff = record.get("diff") or ""
            summary = record.get("summary") or ""
        else:
            desc = getattr(record, "description", "") or ""
            diff = getattr(record, "diff", "") or ""
            summary = getattr(record, "summary", "") or ""
        return "\n".join(part for part in (desc, diff, summary) if part)

    # ------------------------------------------------------------------
    def embed_all(self) -> None:
        """Embed all patch history records using ``SharedVectorService``."""
        cur = self.conn.execute(
            "SELECT id, description, diff, summary, timestamp, enhancement_name FROM patch_history"
        )
        rows = cur.fetchall()
        for pid, desc, diff, summary, ts, enh in rows:
            text = self._compose({"description": desc, "diff": diff, "summary": summary})
            metadata = {"patch_id": pid, "timestamp": ts, "enhancement_name": enh}
            self.svc.vectorise_and_store(
                "text",
                str(pid),
                {"text": text},
                origin_db="patch_history",
                metadata=metadata,
            )


# ----------------------------------------------------------------------
def backfill_patches(path: str | Path | None = None) -> None:
    """Embed existing patch history records."""
    Embedder(path).embed_all()


__all__ = ["Embedder", "backfill_patches"]
