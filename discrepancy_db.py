"""SQLite-backed storage for discrepancy messages with embeddings."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

from vector_service import EmbeddableDBMixin

try:  # pragma: no cover - package and top-level imports
    from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
except Exception:  # pragma: no cover - fallback for tests
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

logger = logging.getLogger(__name__)


@dataclass
class DiscrepancyRecord:
    """Representation of a discrepancy message."""

    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    id: int = 0


class DiscrepancyDB(EmbeddableDBMixin):
    """SQLite-backed storage for discrepancy messages with embeddings."""

    def __init__(
        self,
        path: str | Path = "discrepancies.db",
        *,
        router: DBRouter | None = None,
        vector_index_path: str | Path | None = None,
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "discrepancies", str(path), str(path)
        )
        self.conn = self.router.get_connection("discrepancies")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS discrepancies(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                metadata TEXT,
                ts TEXT
            )
            """
        )
        self.conn.commit()
        index_path = (
            Path(vector_index_path)
            if vector_index_path is not None
            else Path(path).with_suffix(".index")
        )
        meta_path = Path(index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    # ------------------------------------------------------------------
    def _embed_text(self, message: str, meta: Dict[str, Any]) -> str:
        parts = [message]
        for k, v in meta.items():
            if v not in (None, ""):
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def license_text(self, rec: Any) -> str | None:
        if isinstance(rec, DiscrepancyRecord):
            return rec.message
        if isinstance(rec, dict):
            return rec.get("message")
        if isinstance(rec, (int, str)):
            try:
                rid = int(rec)
            except Exception:
                return str(rec)
            row = self.conn.execute(
                "SELECT message FROM discrepancies WHERE id=?", (rid,)
            ).fetchone()
            return row["message"] if row else None
        if isinstance(rec, str):
            return rec
        return None

    def vector(self, rec: Any) -> List[float] | None:
        if isinstance(rec, DiscrepancyRecord):
            text = self._embed_text(rec.message, rec.metadata)
            return self.encode_text(text)
        if isinstance(rec, dict):
            msg = rec.get("message", "")
            meta = {k: v for k, v in rec.items() if k != "message"}
            text = self._embed_text(msg, meta)
            return self.encode_text(text)
        if isinstance(rec, (int, str)):
            try:
                rid = int(rec)
            except Exception:
                return self.encode_text(str(rec))
            row = self.conn.execute(
                "SELECT message, metadata FROM discrepancies WHERE id=?", (rid,)
            ).fetchone()
            if not row:
                return None
            meta = json.loads(row["metadata"] or "{}")
            text = self._embed_text(row["message"], meta)
            return self.encode_text(text)
        if isinstance(rec, str):
            return self.encode_text(rec)
        return None

    # ------------------------------------------------------------------
    def _embed_record_on_write(self, rec_id: int, rec: DiscrepancyRecord) -> None:
        try:
            data = {"message": rec.message, **rec.metadata}
            self.add_embedding(rec_id, data, "discrepancy", source_id=str(rec_id))
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s: %s", rec_id, exc)

    def add(self, rec: DiscrepancyRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO discrepancies(message, metadata, ts) VALUES (?,?,?)",
            (rec.message, json.dumps(rec.metadata), rec.ts),
        )
        self.conn.commit()
        rec.id = int(cur.lastrowid)
        self._embed_record_on_write(rec.id, rec)
        return rec.id

    def get(self, rec_id: int) -> DiscrepancyRecord | None:
        row = self.conn.execute(
            "SELECT id, message, metadata, ts FROM discrepancies WHERE id=?",
            (rec_id,),
        ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata"] or "{}")
        return DiscrepancyRecord(
            message=row["message"], metadata=meta, ts=row["ts"], id=row["id"]
        )

    def update(
        self,
        rec_id: int,
        *,
        message: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        existing = self.get(rec_id)
        if not existing:
            return
        if message is not None:
            existing.message = message
        if metadata is not None:
            existing.metadata = metadata
        self.conn.execute(
            "UPDATE discrepancies SET message=?, metadata=?, ts=? WHERE id=?",
            (existing.message, json.dumps(existing.metadata), existing.ts, rec_id),
        )
        self.conn.commit()
        self._embed_record_on_write(rec_id, existing)

    def delete(self, rec_id: int) -> None:
        self.conn.execute("DELETE FROM discrepancies WHERE id=?", (rec_id,))
        self.conn.commit()
        rid = str(rec_id)
        if rid in self._metadata:
            del self._metadata[rid]
            try:
                self.save_index()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def iter_records(self) -> Iterator[tuple[int, dict[str, Any], str]]:
        cur = self.conn.execute("SELECT id, message, metadata FROM discrepancies")
        for row in cur.fetchall():
            meta = json.loads(row["metadata"] or "{}")
            data = {"message": row["message"], **meta}
            yield row["id"], data, "discrepancy"

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        EmbeddableDBMixin.backfill_embeddings(self)
