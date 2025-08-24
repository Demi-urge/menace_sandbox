"""ChatGPT Enhancement Bot for generating improvement ideas."""

from __future__ import annotations

import json
import os
import sqlite3
import logging
import difflib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Iterator, Sequence, Literal

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from .override_policy import OverridePolicyManager

from .chatgpt_idea_bot import ChatGPTClient
from gpt_memory_interface import GPTMemoryInterface
try:  # pragma: no cover - allow flat imports
    from .memory_aware_gpt_client import ask_with_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_aware_gpt_client import ask_with_memory  # type: ignore
from . import RAISE_ERRORS
try:  # canonical tag constants
    from .log_tags import IMPROVEMENT_PATH, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import IMPROVEMENT_PATH, INSIGHT  # type: ignore
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore
from vector_service import EmbeddableDBMixin
from .unified_event_bus import UnifiedEventBus
from .db_scope import Scope, build_scope_clause

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

try:
    from gensim.summarization import summarize  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    summarize = None  # type: ignore

DEFAULT_DB_PATH = Path(__file__).parent / "enhancements.db"
DB_PATH = Path(os.environ.get("ENHANCEMENT_DB_PATH", str(DEFAULT_DB_PATH)))
FEASIBLE_WORD_LIMIT = int(os.environ.get("MAX_RATIONALE_WORDS", "50"))
DEFAULT_NUM_IDEAS = int(os.environ.get("DEFAULT_ENHANCEMENT_COUNT", "3"))
DEFAULT_FETCH_LIMIT = int(os.environ.get("ENHANCEMENT_FETCH_LIMIT", "50"))
DEFAULT_PROPOSE_RATIO = float(os.environ.get("PROPOSE_SUMMARY_RATIO", "0.3"))


@dataclass
class Enhancement:
    """Representation of an improvement suggestion."""

    idea: str
    rationale: str
    summary: str = ""
    score: float = 0.0
    context: str = ""
    before_code: str = ""
    after_code: str = ""
    timestamp: str = datetime.utcnow().isoformat()
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    type_: str = ""
    assigned_bots: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    cost_estimate: float = 0.0
    category: str = ""
    associated_bots: List[str] = field(default_factory=list)
    triggered_by: str = ""
    model_ids: List[int] = field(default_factory=list)
    bot_ids: List[int] = field(default_factory=list)
    workflow_ids: List[int] = field(default_factory=list)


@dataclass
class EnhancementHistory:
    """Record of an applied enhancement."""

    file_path: str
    original_hash: str
    enhanced_hash: str
    metric_delta: float
    author_bot: str
    ts: str = datetime.utcnow().isoformat()


class EnhancementDB(EmbeddableDBMixin):
    """SQLite storage for enhancement logs with vector embeddings."""

    def __init__(
        self,
        path: Optional[Path] = None,
        override_manager: Optional[OverridePolicyManager] = None,
        *,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "enhancement_embeddings.ann",
        metadata_path: Path | str | None = None,
        embedding_version: int = 1,
        event_bus: UnifiedEventBus | None = None,
        router: "DBRouter | None" = None,
    ) -> None:
        self.path = Path(path) if path else DB_PATH
        self.override_manager = override_manager
        self.router = router or GLOBAL_ROUTER
        if self.router is None:
            self.router = init_db_router("enh", str(self.path), str(self.path))
        self.conn = self.router.get_connection("enhancements")
        self.conn.row_factory = sqlite3.Row
        self._init()
        self.vector_backend = vector_backend  # kept for compatibility
        self.event_bus = event_bus
        if metadata_path is None:
            metadata_path = Path(vector_index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=metadata_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    def _current_menace_id(self, source_menace_id: str | None) -> str:
        return source_menace_id or (
            self.router.menace_id if self.router else ""
        )

    def _connect(self) -> sqlite3.Connection:  # pragma: no cover - simple wrapper
        return self.conn

    def _init(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception("failed to create db directory %s: %s", self.path.parent, exc)
            if RAISE_ERRORS:
                raise
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS enhancements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        idea TEXT,
                        rationale TEXT,
                        summary TEXT,
                        score REAL,
                        timestamp TEXT,
                        context TEXT,
                        before_code TEXT,
                        after_code TEXT,
                        title TEXT,
                        description TEXT,
                        tags TEXT,
                        type TEXT,
                        assigned_bots TEXT,
                        rejection_reason TEXT,
                        cost_estimate REAL,
                        category TEXT,
                        associated_bots TEXT,
                        triggered_by TEXT,
                        source_menace_id TEXT NOT NULL
                    )
                    """
                )
                cols = [
                    r[1]
                    for r in conn.execute(
                        "PRAGMA table_info(enhancements)"
                    ).fetchall()
                ]
                if "before_code" not in cols:
                    conn.execute(
                        "ALTER TABLE enhancements ADD COLUMN before_code TEXT"
                    )
                if "after_code" not in cols:
                    conn.execute(
                        "ALTER TABLE enhancements ADD COLUMN after_code TEXT"
                    )
                if "triggered_by" not in cols:
                    conn.execute(
                        "ALTER TABLE enhancements ADD COLUMN triggered_by TEXT"
                    )
                if "source_menace_id" not in cols:
                    conn.execute(
                        (
                            "ALTER TABLE enhancements ADD COLUMN source_menace_id "
                            "TEXT NOT NULL DEFAULT ''"
                        )
                    )
                idxs = [
                    r[1]
                    for r in conn.execute(
                        "PRAGMA index_list(enhancements)"
                    ).fetchall()
                ]
                if "ix_enhancements_source_menace_id" in idxs:
                    conn.execute(
                        "DROP INDEX ix_enhancements_source_menace_id",
                    )
                if "idx_enhancements_source_menace_id" not in idxs:
                    conn.execute(
                        (
                            "CREATE INDEX idx_enhancements_source_menace_id "
                            "ON enhancements(source_menace_id)"
                        )
                    )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS enhancement_models (
                        enhancement_id INTEGER,
                        model_id INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS enhancement_bots (
                        enhancement_id INTEGER,
                        bot_id INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS enhancement_workflows (
                        enhancement_id INTEGER,
                        workflow_id INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        error_fingerprint TEXT,
                        prompt TEXT,
                        fix TEXT,
                        success INTEGER,
                        ts TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS enhancement_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT,
                        original_hash TEXT,
                        enhanced_hash TEXT,
                        metric_delta REAL,
                        author_bot TEXT,
                        ts TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS enhancement_embeddings (
                        record_id TEXT PRIMARY KEY,
                        kind TEXT
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.exception("database initialization failed: %s", exc)
            if RAISE_ERRORS:
                raise

    def add(self, enh: Enhancement) -> int:
        tags = ",".join(enh.tags)
        assigned = ",".join(enh.assigned_bots)
        assoc = ",".join(enh.associated_bots)
        menace_id = self._current_menace_id(None)
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO enhancements(
                        source_menace_id,
                        idea, rationale, summary, score, timestamp, context,
                        before_code, after_code, title, description, tags, type, assigned_bots,
                        rejection_reason, cost_estimate, category, associated_bots,
                        triggered_by
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        menace_id,
                        enh.idea,
                        enh.rationale,
                        enh.summary,
                        enh.score,
                        enh.timestamp,
                        enh.context,
                        enh.before_code,
                        enh.after_code,
                        enh.title,
                        enh.description,
                        tags,
                        enh.type_,
                        assigned,
                        enh.rejection_reason,
                        enh.cost_estimate,
                        enh.category,
                        assoc,
                        enh.triggered_by,
                    ),
                )
                conn.commit()
                enh_id = int(cur.lastrowid)
        except sqlite3.Error as exc:
            logger.exception("failed to add enhancement: %s", exc)
            if RAISE_ERRORS:
                raise
            return -1
        # generate vector embedding for the newly inserted record
        try:
            self.add_embedding(enh_id, enh, "enhancement", source_id=str(enh_id))
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("failed to add enhancement embedding: %s", exc)
            if RAISE_ERRORS:
                raise
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception:
                logger.exception("failed to publish embedding event")
        return enh_id

    def update(self, enhancement_id: int, enh: Enhancement) -> None:
        tags = ",".join(enh.tags)
        assigned = ",".join(enh.assigned_bots)
        assoc = ",".join(enh.associated_bots)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE enhancements SET
                        idea=?, rationale=?, summary=?, score=?, timestamp=?, context=?,
                        before_code=?, after_code=?, title=?, description=?, tags=?,
                        type=?, assigned_bots=?,
                        rejection_reason=?, cost_estimate=?, category=?,
                        associated_bots=?, triggered_by=?
                    WHERE id=?
                    """,
                    (
                        enh.idea,
                        enh.rationale,
                        enh.summary,
                        enh.score,
                        enh.timestamp,
                        enh.context,
                        enh.before_code,
                        enh.after_code,
                        enh.title,
                        enh.description,
                        tags,
                        enh.type_,
                        assigned,
                        enh.rejection_reason,
                        enh.cost_estimate,
                        enh.category,
                        assoc,
                        enh.triggered_by,
                        enhancement_id,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.exception("failed to update enhancement: %s", exc)
            if RAISE_ERRORS:
                raise
            return
        # update embedding for modified record
        try:
            self.add_embedding(
                enhancement_id,
                enh,
                "enhancement",
                source_id=str(enhancement_id),
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("failed to update enhancement embedding: %s", exc)
            if RAISE_ERRORS:
                raise

    def add_embedding(self, record_id: Any, record: Any, kind: str, *, source_id: str = "") -> None:
        """Store embedding via mixin and mirror minimal metadata in SQLite."""
        super().add_embedding(record_id, record, kind, source_id=source_id)
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO enhancement_embeddings(record_id, kind) VALUES (?, ?)",
                (str(record_id), kind),
            )
            self.conn.commit()
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist embedding metadata for %s", record_id)

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(
        self,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> Iterator[tuple[int, sqlite3.Row, str]]:
        """Yield enhancement rows for embedding backfill."""
        menace_id = self._current_menace_id(source_menace_id)
        clause, params = build_scope_clause("enhancements", Scope(scope), menace_id)
        query = "SELECT * FROM enhancements"
        if clause:
            query += f" {clause}"
        cur = self.conn.execute(query, params)
        for row in cur.fetchall():
            yield row["id"], row, "enhancement"

    def link_model(self, enhancement_id: int, model_id: int) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO enhancement_models (enhancement_id, model_id) VALUES (?, ?)",
                    (enhancement_id, model_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.exception("failed to link model: %s", exc)
            if RAISE_ERRORS:
                raise

    def link_bot(self, enhancement_id: int, bot_id: int) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO enhancement_bots (enhancement_id, bot_id) VALUES (?, ?)",
                    (enhancement_id, bot_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.exception("failed to link bot: %s", exc)
            if RAISE_ERRORS:
                raise

    def link_workflow(self, enhancement_id: int, workflow_id: int) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO enhancement_workflows (enhancement_id, workflow_id) VALUES (?, ?)",
                    (enhancement_id, workflow_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.exception("failed to link workflow: %s", exc)
            if RAISE_ERRORS:
                raise

    def models_for(self, enhancement_id: int) -> List[int]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT model_id FROM enhancement_models WHERE enhancement_id=?",
                    (enhancement_id,),
                ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as exc:
            logger.exception("fetch models failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return []

    def bots_for(self, enhancement_id: int) -> List[int]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT bot_id FROM enhancement_bots WHERE enhancement_id=?",
                    (enhancement_id,),
                ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as exc:
            logger.exception("fetch bots failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return []

    def workflows_for(self, enhancement_id: int) -> List[int]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT workflow_id FROM enhancement_workflows WHERE enhancement_id=?",
                    (enhancement_id,),
                ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as exc:
            logger.exception("fetch workflows failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return []

    def roi_uplift(
        self,
        enhancement_id: int,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> float | None:
        """Return ROI uplift score for ``enhancement_id`` from ``enhancements.score``."""
        menace_id = self._current_menace_id(source_menace_id)
        try:
            with self._connect() as conn:
                clause, params = build_scope_clause(
                    "enhancements", Scope(scope), menace_id
                )
                query = "SELECT score FROM enhancements"
                params = list(params)
                if clause:
                    query += f" {clause} AND id=?"
                else:
                    query += " WHERE id=?"
                params.append(enhancement_id)
                row = conn.execute(query, params).fetchone()
            return float(row["score"]) if row else None
        except sqlite3.Error as exc:
            logger.exception("fetch roi uplift failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return None

    def fetch(
        self,
        limit: int = DEFAULT_FETCH_LIMIT,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[Enhancement]:
        try:
            menace_id = self._current_menace_id(source_menace_id)
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                clause, params = build_scope_clause(
                    "enhancements", Scope(scope), menace_id
                )
                query = "SELECT * FROM enhancements"
                if clause:
                    query += f" {clause}"
                query += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(query, params).fetchall()
            results: List[Enhancement] = []
            for row in rows:
                e_id = row["id"]
                with self._connect() as conn:
                    m_rows = conn.execute(
                        "SELECT model_id FROM enhancement_models WHERE enhancement_id=?",
                        (e_id,),
                    ).fetchall()
                    b_rows = conn.execute(
                        "SELECT bot_id FROM enhancement_bots WHERE enhancement_id=?",
                        (e_id,),
                    ).fetchall()
                    w_rows = conn.execute(
                        "SELECT workflow_id FROM enhancement_workflows WHERE enhancement_id=?",
                        (e_id,),
                    ).fetchall()
                results.append(
                    Enhancement(
                        idea=row["idea"],
                        rationale=row["rationale"],
                        summary=row["summary"],
                        score=row["score"],
                        timestamp=row["timestamp"],
                        context=row["context"],
                        before_code=row["before_code"] or "",
                        after_code=row["after_code"] or "",
                        title=row["title"] or "",
                        description=row["description"] or "",
                        tags=row["tags"].split(",") if row["tags"] else [],
                        type_=row["type"] or "",
                        assigned_bots=(
                            row["assigned_bots"].split(",")
                            if row["assigned_bots"]
                            else []
                        ),
                        rejection_reason=row["rejection_reason"] or "",
                        cost_estimate=row["cost_estimate"] or 0.0,
                        category=row["category"] or "",
                        associated_bots=(
                            row["associated_bots"].split(",")
                            if row["associated_bots"]
                            else []
                        ),
                        triggered_by=row["triggered_by"] or "",
                        model_ids=[r[0] for r in m_rows],
                        bot_ids=[r[0] for r in b_rows],
                        workflow_ids=[r[0] for r in w_rows],
                    )
                )
            return results
        except sqlite3.Error as exc:
            logger.exception("fetch enhancements failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return []

    # --------------------------------------------------------------
    # embedding/search helpers
    def vector(
        self,
        rec: Any,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> list[float] | None:
        """Return an embedding vector for ``rec`` or ``rec`` id."""

        menace_id = self._current_menace_id(source_menace_id)
        # allow passing a record identifier
        if isinstance(rec, (int, str)) and str(rec).isdigit():
            clause, params = build_scope_clause(
                "enhancements", Scope(scope), menace_id
            )
            query = (
                "SELECT before_code, after_code, summary, context FROM enhancements"
            )
            params = list(params)
            if clause:
                query += f" {clause} AND id=?"
            else:
                query += " WHERE id=?"
            params.append(int(rec))
            row = self.conn.execute(query, params).fetchone()
            if not row:
                return None
            rec = row

        # normalise mapping/row objects
        if isinstance(rec, Enhancement):
            before, after, summary, context = (
                rec.before_code,
                rec.after_code,
                rec.summary,
                rec.context,
            )
        else:
            if isinstance(rec, sqlite3.Row):
                rec = dict(rec)
            before = rec.get("before_code", "") if isinstance(rec, dict) else ""
            after = rec.get("after_code", "") if isinstance(rec, dict) else ""
            summary = rec.get("summary", "") if isinstance(rec, dict) else ""
            context = rec.get("context", "") if isinstance(rec, dict) else ""

        parts: List[str] = []
        if before or after:
            diff = "\n".join(
                difflib.unified_diff(
                    (before or "").splitlines(),
                    (after or "").splitlines(),
                    lineterm="",
                )
            )
            if diff:
                parts.append(diff)
        elif context:
            parts.append(context)
        if summary:
            parts.append(summary)

        text = "\n".join(parts).strip()
        if not text:
            return None
        return self._embed(text)

    def _embed(self, text: str) -> list[float] | None:
        """Encode ``text`` using the shared sentence transformer."""
        try:  # pragma: no cover - optional dependency
            return self.encode_text(text)
        except Exception:
            return None

    def search_by_vector(
        self,
        vector: Iterable[float],
        top_k: int = 5,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[Enhancement]:
        menace_id = self._current_menace_id(source_menace_id)
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[Enhancement] = []
        clause, base_params = build_scope_clause(
            "enhancements", Scope(scope), menace_id
        )
        for rec_id, dist in matches:
            query = "SELECT * FROM enhancements"
            params = list(base_params)
            if clause:
                query += f" {clause} AND id=?"
            else:
                query += " WHERE id=?"
            params.append(rec_id)
            row = self.conn.execute(query, params).fetchone()
            if row:
                enh = Enhancement(
                    idea=row["idea"],
                    rationale=row["rationale"],
                    summary=row["summary"],
                    score=row["score"],
                    timestamp=row["timestamp"],
                    context=row["context"],
                    before_code=row["before_code"] or "",
                    after_code=row["after_code"] or "",
                    title=row["title"] or "",
                    description=row["description"] or "",
                    tags=row["tags"].split(",") if row["tags"] else [],
                    type_=row["type"] or "",
                    assigned_bots=(
                        row["assigned_bots"].split(",")
                        if row["assigned_bots"]
                        else []
                    ),
                    rejection_reason=row["rejection_reason"] or "",
                    cost_estimate=row["cost_estimate"] or 0.0,
                    category=row["category"] or "",
                    associated_bots=(
                        row["associated_bots"].split(",")
                        if row["associated_bots"]
                        else []
                    ),
                    triggered_by=row["triggered_by"] or "",
                    model_ids=self.models_for(rec_id),
                    bot_ids=self.bots_for(rec_id),
                    workflow_ids=self.workflows_for(rec_id),
                )
                setattr(enh, "id", rec_id)
                setattr(enh, "_distance", dist)
                results.append(enh)
        return results

    def add_prompt_history(
        self, fingerprint: str, prompt: str, fix: str = "", success: bool = False
    ) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO prompt_history(
                        error_fingerprint, prompt, fix, success, ts
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        fingerprint,
                        prompt,
                        fix,
                        int(success),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
            if self.override_manager:
                try:
                    self.override_manager.record_fix(fix or fingerprint, success)
                except Exception as exc:  # pragma: no cover - best effort logging
                    logger.exception("record fix failed: %s", exc)
                    if RAISE_ERRORS:
                        raise
            return int(cur.lastrowid)
        except sqlite3.Error as exc:
            logger.exception("failed to record prompt history: %s", exc)
            if RAISE_ERRORS:
                raise
            return -1

    def prompt_history(self, fingerprint: str) -> List[dict[str, object]]:
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM prompt_history WHERE error_fingerprint=? ORDER BY id DESC",
                    (fingerprint,),
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.exception("failed to fetch prompt history: %s", exc)
            if RAISE_ERRORS:
                raise
            return []

    # ------------------------------------------------------------------
    # Enhancement history helpers
    # ------------------------------------------------------------------

    def record_history(self, rec: EnhancementHistory) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO enhancement_history(
                        file_path, original_hash, enhanced_hash, metric_delta, author_bot, ts
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    (
                        rec.file_path,
                        rec.original_hash,
                        rec.enhanced_hash,
                        rec.metric_delta,
                        rec.author_bot,
                        rec.ts,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)
        except sqlite3.Error as exc:
            logger.exception("failed to record enhancement history: %s", exc)
            if RAISE_ERRORS:
                raise
            return -1

    def history(self, limit: int = DEFAULT_FETCH_LIMIT) -> List[EnhancementHistory]:
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT file_path, original_hash, enhanced_hash, metric_delta, author_bot, ts "
                    "FROM enhancement_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [EnhancementHistory(**dict(r)) for r in rows]
        except sqlite3.Error as exc:
            logger.exception("failed to fetch enhancement history: %s", exc)
            if RAISE_ERRORS:
                raise
            return []


DEFAULT_SUMMARY_RATIO = float(os.environ.get("SUMMARY_RATIO", "0.2"))


def summarise_text(text: str, ratio: float = DEFAULT_SUMMARY_RATIO) -> str:
    """Summarise text using gensim if available."""
    text = text.strip()
    if not text:
        return ""
    try:
        ratio_val = float(ratio)
    except (TypeError, ValueError):
        logger.error("invalid summary ratio %s; using default", ratio)
        ratio_val = DEFAULT_SUMMARY_RATIO
    ratio_val = max(0.0, min(1.0, ratio_val))
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) <= 1:
        return text
    if summarize:
        try:
            result = summarize(text, ratio=ratio_val)
            if result:
                return result
        except Exception as exc:
            logger.exception("text summary failed: %s", exc)
            if RAISE_ERRORS:
                raise
    count = max(1, int(len(sentences) * ratio_val))
    return ". ".join(sentences[:count]) + "."


def parse_enhancements(data: dict[str, object]) -> List[Enhancement]:
    """Parse enhancements from ChatGPT JSON response."""
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("payload is not a list")
    except Exception as exc:
        logger.exception("failed to parse enhancements: %s", exc)
        return []
    enhancements: List[Enhancement] = []
    for item in items:
        try:
            enhancements.append(
                Enhancement(
                    idea=str(item.get("idea", "")),
                    rationale=str(item.get("rationale", "")),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("invalid enhancement item %s: %s", item, exc)
    return enhancements


class ChatGPTEnhancementBot:
    """Generate and store improvement ideas via ChatGPT."""
    def __init__(
        self,
        client: ChatGPTClient,
        db: Optional[EnhancementDB] = None,
        override_manager: Optional[OverridePolicyManager] = None,
        gpt_memory: GPTMemoryInterface | None = GPT_MEMORY_MANAGER,
    ) -> None:
        self.override_manager = override_manager
        self.client = client
        self.db = db or EnhancementDB(override_manager=override_manager)
        self.gpt_memory = gpt_memory
        if getattr(self.client, "gpt_memory", None) is None:
            try:
                self.client.gpt_memory = self.gpt_memory
            except Exception:
                logger.debug("failed to attach gpt_memory to client", exc_info=True)

    def _feasible(self, enh: Enhancement) -> bool:
        return len(enh.rationale.split()) < FEASIBLE_WORD_LIMIT

    def propose(
        self,
        instruction: str,
        num_ideas: int = DEFAULT_NUM_IDEAS,
        context: str = "",
        ratio: float = DEFAULT_PROPOSE_RATIO,
        *,
        tags: Sequence[str] | None = None,
    ) -> List[Enhancement]:
        prompt = (
            f"{instruction} Provide {num_ideas} enhancement ideas as a JSON list with"
            " fields idea and rationale."
        )
        logger.debug("sending prompt to ChatGPT: %s", prompt)
        try:
            base_tags = [IMPROVEMENT_PATH, INSIGHT]
            if tags:
                base_tags.extend(tags)
            data = ask_with_memory(
                self.client,
                "chatgpt_enhancement_bot.propose",
                prompt,
                memory=self.gpt_memory,
                tags=base_tags,
            )
        except Exception as exc:
            logger.exception("chatgpt request failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return []
        ideas = parse_enhancements(data)
        results: List[Enhancement] = []
        for enh in ideas:
            if not self._feasible(enh):
                logger.debug("enhancement discarded, not feasible: %s", enh.idea)
                continue
            enh.summary = summarise_text(enh.rationale, ratio=ratio)
            enh.context = context
            enh_id = self.db.add(enh)
            logger.debug("stored enhancement %s", enh_id)
            results.append(enh)
        return results


__all__ = [
    "Enhancement",
    "EnhancementHistory",
    "EnhancementDB",
    "ChatGPTEnhancementBot",
    "summarise_text",
    "parse_enhancements",
]
