"""SQLite storage for static information with embedding support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List, Sequence, Iterator
from pathlib import Path
import json
import sqlite3
import logging

from menace.embeddable_db_mixin import EmbeddableDBMixin


logger = logging.getLogger(__name__)


def _serialize_keywords(keywords: Iterable[str]) -> str:
    return ",".join(keywords)


def _deserialize_keywords(val: str) -> List[str]:
    return [v for v in val.split(",") if v]


@dataclass
class InformationRecord:
    data_type: str
    source_url: str = ""
    date_collected: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    summary: str = ""
    validated: bool = False
    validation_notes: str = ""
    keywords: List[str] = field(default_factory=list)
    data_depth_score: float = 0.0
    info_id: int = 0


class InformationDB(EmbeddableDBMixin):
    """SQLite-backed store for static information with embeddings."""

    def __init__(
        self,
        path: str = "information.db",
        *,
        vector_index_path: str = "information_embeddings.index",
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS information(
                info_id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT,
                source_url TEXT,
                date_collected TEXT,
                summary TEXT,
                validated INTEGER,
                validation_notes TEXT,
                keywords TEXT,
                data_depth_score REAL
            )
            """,
        )
        self.conn.commit()
        meta_path = Path(vector_index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    # ------------------------------------------------------------------
    def _flatten_fields(self, data: dict[str, Any]) -> List[str]:
        pairs: List[str] = []

        def _walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    _walk(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(value, (list, tuple, set)):
                for v in value:
                    _walk(prefix, v)
            else:
                if value not in (None, ""):
                    pairs.append(f"{prefix}={value}")

        _walk("", data)
        return pairs

    def _embed_text(self, rec: InformationRecord | dict[str, Any]) -> str:
        if isinstance(rec, InformationRecord):
            data = {
                "data_type": rec.data_type,
                "source_url": rec.source_url,
                "summary": rec.summary,
                "validated": rec.validated,
                "validation_notes": rec.validation_notes,
                "keywords": rec.keywords,
            }
        else:
            data = {
                "data_type": rec.get("data_type", ""),
                "source_url": rec.get("source_url", ""),
                "summary": rec.get("summary", ""),
                "validated": rec.get("validated", ""),
                "validation_notes": rec.get("validation_notes", ""),
                "keywords": _deserialize_keywords(rec.get("keywords", "")),
            }
        return " ".join(self._flatten_fields(data))

    # ------------------------------------------------------------------
    def _embed_record_on_write(
        self, info_id: int, rec: InformationRecord | dict[str, Any]
    ) -> None:
        """Best-effort embedding hook for inserts and updates."""

        try:
            self.add_embedding(info_id, rec, "info", source_id=str(info_id))
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s: %s", info_id, exc)

    def add(self, rec: InformationRecord) -> int:
        keywords = _serialize_keywords(rec.keywords)
        cur = self.conn.execute(
            """
            INSERT INTO information(data_type, source_url, date_collected, summary, validated,
                                     validation_notes, keywords, data_depth_score)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                rec.data_type,
                rec.source_url,
                rec.date_collected,
                rec.summary,
                int(rec.validated),
                rec.validation_notes,
                keywords,
                rec.data_depth_score,
            ),
        )
        self.conn.commit()
        rec.info_id = int(cur.lastrowid)
        self._embed_record_on_write(rec.info_id, rec)
        return rec.info_id

    def update(self, info_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        params = list(fields.values()) + [info_id]
        self.conn.execute(f"UPDATE information SET {sets} WHERE info_id=?", params)
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM information WHERE info_id=?", (info_id,)
        ).fetchone()
        if row:
            self._embed_record_on_write(info_id, dict(row))

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(self) -> Iterator[tuple[int, dict[str, Any], str]]:
        """Yield information rows for embedding backfill."""
        cur = self.conn.execute("SELECT * FROM information")
        for row in cur.fetchall():
            yield row["info_id"], dict(row), "info"

    # ------------------------------------------------------------------
    def vector(self, rec: Any) -> List[float] | None:
        if isinstance(rec, (int, str)):
            rid = str(rec)
            meta = self._metadata.get(rid)
            if meta and "vector" in meta:
                return meta["vector"]
            try:
                rec_id = int(rec)
            except (TypeError, ValueError):
                return None
            row = self.conn.execute(
                "SELECT * FROM information WHERE info_id=?", (rec_id,)
            ).fetchone()
            if not row:
                return None
            text = self._embed_text(dict(row))
            return self._embed(text) if text else None
        text = self._embed_text(rec)
        return self._embed(text) if text else None

    def _embed(self, text: str) -> List[float]:
        """Encode ``text`` to a vector (overridable for tests)."""
        return self.encode_text(text)

    def search_by_vector(self, vector: Sequence[float], top_k: int = 5) -> List[dict[str, Any]]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[dict[str, Any]] = []
        for rec_id, dist in matches:
            row = self.conn.execute(
                "SELECT * FROM information WHERE info_id=?", (rec_id,)
            ).fetchone()
            if row:
                rec = dict(row)
                rec["_distance"] = dist
                rec["keywords"] = _deserialize_keywords(rec.get("keywords", ""))
                results.append(rec)
        return results
