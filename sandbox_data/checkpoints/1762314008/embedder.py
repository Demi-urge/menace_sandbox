from __future__ import annotations

"""Embed patch history records using :class:`SharedVectorService`."""

from pathlib import Path
from typing import Any
import logging

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
    def _latest_patch_id(self) -> int:
        """Return the highest ``patch_id`` already embedded."""
        store = getattr(self.svc, "vector_store", None)
        max_id = 0
        if store is not None and hasattr(store, "meta"):
            for meta in getattr(store, "meta", []):
                pid = (meta.get("metadata") or {}).get("patch_id")
                if isinstance(pid, int) and pid > max_id:
                    max_id = pid
        return max_id

    def embed_since(self, last_id: int) -> int:
        """Embed patches with ``id`` greater than ``last_id``.

        Returns the latest successfully embedded ``patch_id``.
        """
        cur = self.conn.execute(
            "SELECT id, description, diff, summary, timestamp, enhancement_name FROM patch_history WHERE id>? ORDER BY id",
            (last_id,),
        )
        latest = last_id
        logger = logging.getLogger(__name__)
        for pid, desc, diff, summary, ts, enh in cur:
            text = self._compose({"description": desc, "diff": diff, "summary": summary})
            metadata = {"patch_id": pid, "timestamp": ts, "enhancement_name": enh}
            try:
                self.svc.vectorise_and_store(
                    "text",
                    str(pid),
                    {"text": text},
                    origin_db="patch_history",
                    metadata=metadata,
                )
                latest = pid
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("failed to embed patch %s: %s", pid, exc)
        return latest

    # ------------------------------------------------------------------
    def embed_all(self) -> int:
        """Embed new patch history records using ``SharedVectorService``."""
        last_id = self._latest_patch_id()
        return self.embed_since(last_id)


# ----------------------------------------------------------------------
def backfill_patches(path: str | Path | None = None) -> None:
    """Embed existing patch history records."""
    Embedder(path).embed_all()


__all__ = ["Embedder", "backfill_patches"]
