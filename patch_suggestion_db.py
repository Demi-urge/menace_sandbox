from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, TYPE_CHECKING
import logging
import threading
from filelock import FileLock

from db_router import init_db_router

from analysis.semantic_diff_filter import find_semantic_risks
from patch_safety import PatchSafety
try:  # pragma: no cover - allow flat imports
    from vector_service import EmbeddableDBMixin  # type: ignore
    if EmbeddableDBMixin is object or not hasattr(EmbeddableDBMixin, "add_embedding"):
        raise ImportError
except Exception:  # pragma: no cover - provide lightweight fallback
    class EmbeddableDBMixin:  # type: ignore
        """Minimal embedding mixin used when vector_service is unavailable."""

        def __init__(self, *a: Any, **k: Any) -> None:
            self._metadata: Dict[str, Dict[str, Any]] = {}

        def encode_text(self, text: str) -> List[float]:  # pragma: no cover - dummy
            return []

        def add_embedding(
            self,
            record_id: Any,
            record: Any,
            kind: str,
            *,
            source_id: str = "",
        ) -> None:
            vec = self.vector(record)
            if vec is None:
                return
            self._metadata[str(record_id)] = {"vector": vec}

        def search_by_vector(
            self, vector: List[float], top_k: int = 5
        ) -> List[tuple[str, float]]:
            results: List[tuple[str, float]] = []
            for rid, meta in self._metadata.items():
                vec = meta.get("vector", [])
                score = sum(a * b for a, b in zip(vector, vec))
                results.append((rid, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        def backfill_embeddings(self, batch_size: int = 100) -> None:
            for rec_id, rec, kind in self.iter_records():
                if str(rec_id) not in self._metadata:
                    self.add_embedding(rec_id, rec, kind)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from enhancement_classifier import EnhancementSuggestion

logger = logging.getLogger(__name__)


@dataclass
class SuggestionRecord:
    module: str
    description: str
    score: float = 0.0
    rationale: str = ""
    patch_count: int = 0
    module_id: str = ""
    ts: str = datetime.utcnow().isoformat()


@dataclass
class EnhancementSuggestionRecord:
    module: str
    score: float
    rationale: str
    first_seen: str
    last_seen: str
    occurrences: int = 1


class PatchSuggestionDB(EmbeddableDBMixin):
    """Store successful patch descriptions per module with embeddings."""

    def __init__(
        self,
        path: Path | str = "suggestions.db",
        *,
        semantic_threshold: float = 0.5,
        safety: PatchSafety | None = None,
        vector_index_path: str | Path | None = None,
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._file_lock = FileLock(str(self.path) + ".lock")
        self._semantic_threshold = semantic_threshold
        self._safety = safety
        self.router = init_db_router(
            "patch_suggestion_db", str(self.path), str(self.path)
        )
        self.conn = self.router.get_connection("patches")
        with self._file_lock:
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS suggestions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT,
                description TEXT,
                score REAL DEFAULT 0,
                rationale TEXT,
                patch_count INTEGER DEFAULT 0,
                module_id TEXT,
                ts TEXT
            )
            """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_suggestions_module ON suggestions(module)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_suggestions_score ON suggestions(score)"
            )
            try:
                self.conn.execute(
                    "ALTER TABLE suggestions ADD COLUMN patch_count INTEGER DEFAULT 0"
                )
            except Exception:
                pass
            try:
                self.conn.execute(
                    "ALTER TABLE suggestions ADD COLUMN module_id TEXT"
                )
            except Exception:
                pass
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT,
                description TEXT,
                accepted INTEGER,
                reason TEXT,
                ts TEXT
            )
            """
            )
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS failed_strategies(
                tag TEXT PRIMARY KEY,
                ts TEXT
            )
            """
            )
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS enhancement_suggestions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT,
                score REAL,
                rationale TEXT,
                first_seen TEXT,
                last_seen TEXT,
                occurrences INTEGER DEFAULT 1
            )
            """
            )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_enh_sugg_unique ON enhancement_suggestions(module, rationale)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_enh_sugg_score ON enhancement_suggestions(score)"
            )
            self.conn.commit()

        index_path = (
            Path(vector_index_path)
            if vector_index_path is not None
            else self.path.with_suffix(".index")
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
    def _embed_text(
        self,
        module: str,
        description: str,
        rationale: str,
        patch_count: int,
        module_id: str,
    ) -> str:
        return (
            f"module={module} module_id={module_id} patches={patch_count} "
            f"description={description} rationale={rationale}"
        )

    def license_text(self, rec: Any) -> str | None:
        if isinstance(rec, SuggestionRecord):
            return rec.description
        if isinstance(rec, dict):
            return rec.get("description")
        return None

    def vector(self, rec: Any) -> List[float] | None:
        if isinstance(rec, SuggestionRecord):
            text = self._embed_text(
                rec.module,
                rec.description,
                rec.rationale,
                rec.patch_count,
                rec.module_id,
            )
            return self.encode_text(text)
        if isinstance(rec, dict):
            text = self._embed_text(
                rec.get("module", ""),
                rec.get("description", ""),
                rec.get("rationale", ""),
                int(rec.get("patch_count", 0)),
                rec.get("module_id", ""),
            )
            return self.encode_text(text)
        return None

    def _embed_record_on_write(self, rec_id: int, rec: SuggestionRecord) -> None:
        try:
            self.add_embedding(rec_id, rec, "suggestion", source_id=str(rec_id))
        except Exception:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s", rec_id)

    def add(self, rec: SuggestionRecord) -> int:
        if self._has_similar(rec.module, rec.rationale or rec.description, rec.score):
            return 0
        risks = find_semantic_risks(
            rec.description.splitlines(), threshold=self._semantic_threshold
        )
        if risks:
            self.log_decision(
                rec.module, rec.description, False, risks[0][1], rec.ts
            )
            raise ValueError(f"unsafe suggestion: {risks[0][1]}")
        if self._safety:
            passed, *_ = self._safety.evaluate(
                {"category": rec.description, "module": rec.module},
                {"category": rec.description, "module": rec.module},
            )
            if not passed:
                self.log_decision(
                    rec.module, rec.description, False, "failure similarity", rec.ts
                )
                raise ValueError("unsafe suggestion: failure similarity")
        with self._file_lock:
            with self._lock:
                cur = self.conn.execute(
                    "INSERT INTO suggestions(module, description, score, rationale, "
                    "patch_count, module_id, ts) VALUES(?,?,?,?,?,?,?)",
                    (
                        rec.module,
                        rec.description,
                        rec.score,
                        rec.rationale,
                        rec.patch_count,
                        rec.module_id,
                        rec.ts,
                    ),
                )
                self.conn.commit()
                rec_id = int(cur.lastrowid)
        self.log_decision(rec.module, rec.description, True, "", rec.ts)
        self._embed_record_on_write(rec_id, rec)
        return rec_id

    def log_decision(
        self,
        module: str,
        description: str,
        accepted: bool,
        reason: str,
        ts: str | None = None,
    ) -> None:
        ts = ts or datetime.utcnow().isoformat()
        with self._file_lock:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO decisions(module, description, accepted, reason, ts) "
                    "VALUES(?,?,?,?,?)",
                    (module, description, int(accepted), reason, ts),
                )
                self.conn.commit()

    def history(self, module: str, limit: int = 10) -> List[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT description FROM suggestions WHERE module=? ORDER BY id DESC LIMIT ?",
                (module, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def best_match(self, module: str) -> Optional[str]:
        past = self.history(module, limit=20)
        if not past:
            return None
        desc, _ = Counter(past).most_common(1)[0]
        return desc

    def _has_similar(
        self, module: str, rationale: str, score: float, tolerance: float = 0.1
    ) -> bool:
        with self._lock:
            rows = self.conn.execute(
                "SELECT score FROM suggestions WHERE module=? AND rationale=?",
                (module, rationale),
            ).fetchall()
        for r in rows:
            existing = r[0] or 0.0
            if abs(existing - score) <= tolerance:
                return True
        return False

    # ------------------------------------------------------------------
    def queue_suggestions(self, suggestions: Iterable["EnhancementSuggestion"]) -> None:
        """Store scored enhancement suggestions for later retrieval."""
        for sugg in suggestions:
            try:
                module = getattr(sugg, "path", "")
                rationale = getattr(sugg, "rationale", "")
                score = float(getattr(sugg, "score", 0.0))
                patch_count = int(getattr(sugg, "patch_count", 0))
                module_id = getattr(sugg, "module_id", "")
                if self._has_similar(module, rationale, score):
                    continue
                rec = SuggestionRecord(
                    module=module,
                    description=rationale,
                    score=score,
                    rationale=rationale,
                    patch_count=patch_count,
                    module_id=module_id,
                )
                self.add(rec)
            except Exception:  # pragma: no cover - best effort
                logger.exception(
                    "failed queueing suggestion for %s", getattr(sugg, "path", "")
                )

    def upsert_enhancement_suggestion(
        self, module: str, score: float, rationale: str
    ) -> None:
        """Insert or update an enhancement suggestion occurrence."""
        ts = datetime.utcnow().isoformat()
        with self._file_lock:
            with self._lock:
                self.conn.execute(
                    """
                INSERT INTO enhancement_suggestions(module, score, rationale, first_seen, last_seen, occurrences)
                VALUES(?,?,?,?,?,1)
                ON CONFLICT(module, rationale) DO UPDATE SET
                    score=excluded.score,
                    last_seen=excluded.last_seen,
                    occurrences=enhancement_suggestions.occurrences + 1
                """,
                    (module, score, rationale, ts, ts),
                )
                self.conn.commit()

    def queue_enhancement_suggestions(
        self, suggestions: Iterable["EnhancementSuggestion"]
    ) -> None:
        """Upsert enhancement suggestions, counting occurrences."""
        for sugg in suggestions:
            try:
                module = getattr(sugg, "path", "")
                rationale = getattr(sugg, "rationale", "")
                score = float(getattr(sugg, "score", 0.0))
                self.upsert_enhancement_suggestion(module, score, rationale)
            except Exception:  # pragma: no cover - best effort
                logger.exception(
                    "failed upserting enhancement suggestion for %s",
                    getattr(sugg, "path", ""),
                )

    def fetch_top_enhancement_suggestions(
        self, limit: int = 10
    ) -> List[EnhancementSuggestionRecord]:
        """Return and remove top scored enhancement suggestions."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, module, score, rationale, first_seen, last_seen, occurrences "
                "FROM enhancement_suggestions ORDER BY score DESC LIMIT ?",
                (limit,),
            ).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                q_marks = ",".join("?" for _ in ids)
                self.conn.execute(
                    f"DELETE FROM enhancement_suggestions WHERE id IN ({q_marks})",
                    ids,
                )
                self.conn.commit()
        return [
            EnhancementSuggestionRecord(
                module=r[1],
                score=r[2] or 0.0,
                rationale=r[3] or "",
                first_seen=r[4],
                last_seen=r[5],
                occurrences=r[6] or 0,
            )
            for r in rows
        ]

    def top_suggestions(self, limit: int = 10) -> List[SuggestionRecord]:
        """Return top scored suggestions ordered by descending score."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT module, description, score, rationale, patch_count, "
                "module_id, ts FROM suggestions ORDER BY score DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            SuggestionRecord(
                module=r[0],
                description=r[1],
                score=r[2] or 0.0,
                rationale=r[3] or "",
                patch_count=r[4] or 0,
                module_id=r[5] or "",
                ts=r[6],
            )
            for r in rows
        ]

    def queued_suggestions(self) -> List[SuggestionRecord]:
        """Return all stored suggestions ordered by insertion."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT module, description, score, rationale, patch_count, "
                "module_id, ts FROM suggestions ORDER BY id"
            ).fetchall()
        return [
            SuggestionRecord(
                module=r[0],
                description=r[1],
                score=r[2] or 0.0,
                rationale=r[3] or "",
                patch_count=r[4] or 0,
                module_id=r[5] or "",
                ts=r[6],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    def add_failed_strategy(self, tag: str) -> None:
        """Persist a failed strategy tag for future exclusion."""
        ts = datetime.utcnow().isoformat()
        with self._file_lock:
            with self._lock:
                self.conn.execute(
                    "INSERT OR IGNORE INTO failed_strategies(tag, ts) VALUES(?,?)",
                    (tag, ts),
                )
                self.conn.commit()

    def failed_strategy_tags(self) -> List[str]:
        """Return all recorded failed strategy tags."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT tag FROM failed_strategies"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    def iter_records(self) -> Iterator[tuple[int, Dict[str, str], str]]:
        cur = self.conn.execute(
            "SELECT id, module, description, rationale, patch_count, module_id FROM suggestions"
        )
        for row in cur.fetchall():
            yield row[0], {
                "module": row[1],
                "description": row[2],
                "rationale": row[3],
                "patch_count": row[4],
                "module_id": row[5],
            }, "suggestion"

    def to_vector_dict(self, rec: SuggestionRecord | Dict[str, str]) -> Dict[str, str]:
        if isinstance(rec, SuggestionRecord):
            return {
                "module": rec.module,
                "description": rec.description,
                "rationale": rec.rationale,
                "patch_count": rec.patch_count,
                "module_id": rec.module_id,
            }
        return {
            "module": rec.get("module", ""),
            "description": rec.get("description", ""),
            "rationale": rec.get("rationale", ""),
            "patch_count": int(rec.get("patch_count", 0)),
            "module_id": rec.get("module_id", ""),
        }

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        EmbeddableDBMixin.backfill_embeddings(self)


__all__ = [
    "PatchSuggestionDB",
    "SuggestionRecord",
    "EnhancementSuggestionRecord",
]
