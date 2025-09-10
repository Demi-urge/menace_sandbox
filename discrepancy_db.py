"""SQLite-backed storage for discrepancy messages with embeddings."""

from __future__ import annotations

import json
import logging
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal

from vector_service import EmbeddableDBMixin

try:  # pragma: no cover - package and top-level imports
    from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
    from .scope_utils import build_scope_clause
except Exception:  # pragma: no cover - fallback for tests
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
    from scope_utils import build_scope_clause

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

    DB_FILE = "discrepancies.db"

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
                ts TEXT,
                source_menace_id TEXT NOT NULL,
                confidence REAL,
                outcome_score REAL
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(discrepancies)").fetchall()]
        if "source_menace_id" not in cols:
            self.conn.execute(
                "ALTER TABLE discrepancies ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
            )
            self.conn.execute(
                "UPDATE discrepancies SET source_menace_id='' WHERE source_menace_id IS NULL"
            )
        if "confidence" not in cols:
            self.conn.execute(
                "ALTER TABLE discrepancies ADD COLUMN confidence REAL"
            )
        if "outcome_score" not in cols:
            self.conn.execute(
                "ALTER TABLE discrepancies ADD COLUMN outcome_score REAL"
            )
        # migrate existing JSON data into the new columns
        self.conn.execute(
            "UPDATE discrepancies SET confidence=json_extract(metadata,'$.confidence') "
            "WHERE confidence IS NULL"
        )
        self.conn.execute(
            "UPDATE discrepancies SET outcome_score=json_extract(metadata,'$.outcome_score') "
            "WHERE outcome_score IS NULL"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_discrepancies_source_menace_id "
            "ON discrepancies(source_menace_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_discrepancies_confidence "
            "ON discrepancies(confidence)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_discrepancies_outcome_score "
            "ON discrepancies(outcome_score)"
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

    def _current_menace_id(self, source_menace_id: str | None) -> str:
        return source_menace_id or (
            self.router.menace_id if self.router else os.getenv("MENACE_ID", "")
        )

    def license_text(
        self,
        rec: Any,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> str | None:
        if isinstance(rec, DiscrepancyRecord):
            return rec.message
        if isinstance(rec, dict):
            return rec.get("message")
        if isinstance(rec, (int, str)):
            try:
                rid = int(rec)
            except Exception:
                return str(rec)
            menace_id = self._current_menace_id(source_menace_id)
            clause, params = build_scope_clause("discrepancies", scope, menace_id)
            params = list(params)
            query = "SELECT message FROM discrepancies"
            if clause:
                query += f" WHERE {clause} AND discrepancies.id=?"
            else:
                query += " WHERE discrepancies.id=?"
            params.append(rid)
            row = self.conn.execute(query, params).fetchone()
            return row["message"] if row else None
        if isinstance(rec, str):
            return rec
        return None

    def vector(
        self,
        rec: Any,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[float] | None:
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
            menace_id = self._current_menace_id(source_menace_id)
            clause, params = build_scope_clause("discrepancies", scope, menace_id)
            params = list(params)
            query = "SELECT message, metadata FROM discrepancies"
            if clause:
                query += f" WHERE {clause} AND discrepancies.id=?"
            else:
                query += " WHERE discrepancies.id=?"
            params.append(rid)
            row = self.conn.execute(query, params).fetchone()
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

    def add(self, rec: DiscrepancyRecord, *, source_menace_id: str | None = None) -> int:
        menace_id = self._current_menace_id(source_menace_id)
        confidence = rec.metadata.get("confidence")
        outcome = rec.metadata.get("outcome_score")
        cur = self.conn.execute(
            "INSERT INTO discrepancies("
            "source_menace_id, message, metadata, ts, confidence, outcome_score)"
            " VALUES (?,?,?,?,?,?)",
            (
                menace_id,
                rec.message,
                json.dumps(rec.metadata),
                rec.ts,
                confidence,
                outcome,
            ),
        )
        self.conn.commit()
        rec.id = int(cur.lastrowid)
        self._embed_record_on_write(rec.id, rec)
        return rec.id

    def get(
        self,
        rec_id: int,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> DiscrepancyRecord | None:
        menace_id = self._current_menace_id(source_menace_id)
        clause, params = build_scope_clause("discrepancies", scope, menace_id)
        params = list(params)
        query = "SELECT id, message, metadata, ts FROM discrepancies"
        if clause:
            query += f" WHERE {clause} AND discrepancies.id=?"
        else:
            query += " WHERE discrepancies.id=?"
        params.append(rec_id)
        row = self.conn.execute(query, params).fetchone()
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
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> None:
        existing = self.get(
            rec_id,
            source_menace_id=source_menace_id,
            scope=scope,
        )
        if not existing:
            return
        if message is not None:
            existing.message = message
        if metadata is not None:
            existing.metadata = metadata
        menace_id = self._current_menace_id(source_menace_id)
        query = (
            "UPDATE discrepancies SET message=?, metadata=?, ts=?, "
            "confidence=?, outcome_score=?"
        )
        confidence = existing.metadata.get("confidence")
        outcome = existing.metadata.get("outcome_score")
        params: list[Any] = [
            existing.message,
            json.dumps(existing.metadata),
            existing.ts,
            confidence,
            outcome,
        ]
        clause, scope_params = build_scope_clause("discrepancies", scope, menace_id)
        params.extend(scope_params)
        if clause:
            query += f" WHERE {clause} AND discrepancies.id=?"
        else:
            query += " WHERE discrepancies.id=?"
        params.append(rec_id)
        self.conn.execute(query, params)
        self.conn.commit()
        self._embed_record_on_write(rec_id, existing)

    def delete(
        self,
        rec_id: int,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> None:
        menace_id = self._current_menace_id(source_menace_id)
        query = "DELETE FROM discrepancies"
        clause, params = build_scope_clause("discrepancies", scope, menace_id)
        params = list(params)
        if clause:
            query += f" WHERE {clause} AND discrepancies.id=?"
        else:
            query += " WHERE discrepancies.id=?"
        params.append(rec_id)
        self.conn.execute(query, params)
        self.conn.commit()
        rid = str(rec_id)
        if rid in self._metadata:
            del self._metadata[rid]
            try:
                self.save_index()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def iter_records(
        self,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> Iterator[tuple[int, dict[str, Any], str]]:
        menace_id = self._current_menace_id(source_menace_id)
        query = "SELECT id, message, metadata FROM discrepancies"
        clause, params = build_scope_clause("discrepancies", scope, menace_id)
        params = list(params)
        if clause:
            query += f" WHERE {clause}"
        cur = self.conn.execute(query, params)
        for row in cur.fetchall():
            meta = json.loads(row["metadata"] or "{}")
            data = {"message": row["message"], **meta}
            yield row["id"], data, "discrepancy"

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        EmbeddableDBMixin.backfill_embeddings(self)
