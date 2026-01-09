"""Research data models and persistence helpers."""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING

if __package__ in (None, ""):
    from auto_link import auto_link
    from chatgpt_research_bot import summarise_text
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
    from unified_event_bus import UnifiedEventBus
else:
    from .auto_link import auto_link
    from .chatgpt_research_bot import summarise_text
    from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
    from .unified_event_bus import UnifiedEventBus
from vector_service import EmbeddableDBMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    if __package__ in (None, ""):
        from menace_db import MenaceDB
        from text_research_bot import TextResearchBot
        from video_research_bot import VideoResearchBot
    else:
        from .menace_db import MenaceDB
        from .text_research_bot import TextResearchBot
        from .video_research_bot import VideoResearchBot

logger = logging.getLogger(__name__)


@dataclass
class ResearchItem:
    """A unit of collected research."""

    topic: str
    content: str
    timestamp: float
    title: str = ""
    tags: List[str] = field(default_factory=list)
    item_id: int = 0
    category: str = ""
    type_: str = ""
    associated_bots: List[str] = field(default_factory=list)
    associated_errors: List[str] = field(default_factory=list)
    performance_data: str = ""
    categories: List[str] = field(default_factory=list)
    quality: float = 0.0
    summary: str = ""
    model_id: int = 0
    contrarian_id: int = 0
    data_depth: float = 0.0
    source_url: str = ""
    notes: str = ""
    energy: int = 1
    corroboration_count: int = 0
    embedding: Optional[List[float]] = None


class InfoDB(EmbeddableDBMixin):
    """SQLite storage for collected research."""

    DB_FILE = "information.db"

    def __init__(
        self,
        path: Path = Path("information.db"),
        current_model_id: int = 0,
        current_contrarian_id: int = 0,
        text_bot: Optional["TextResearchBot"] = None,
        video_bot: Optional["VideoResearchBot"] = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        menace_db: "MenaceDB" | None = None,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "information_embeddings.index",
        embedding_version: int = 1,
        router: DBRouter | None = None,
        run_migrations: bool = True,
        apply_nonessential_migrations: bool = True,
        batch_migrations: bool = False,
        bootstrap_mode: bool = False,
        migration_timeout: float | None = None,
        non_blocking_migrations: bool | None = None,
    ) -> None:
        self.path = path
        self.event_bus = event_bus
        self.current_model_id = current_model_id
        self.current_contrarian_id = current_contrarian_id
        self.text_bot = text_bot
        self.video_bot = video_bot
        self.menace_db = menace_db
        self.has_fts = False
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "information", str(path), str(path)
        )
        self.conn = self.router.get_connection("information")
        self.conn.row_factory = sqlite3.Row
        self._schema_initialised = False
        self._schema_ready_cached = bool(bootstrap_mode)
        self._apply_nonessential_migrations = apply_nonessential_migrations
        self._batch_migrations = batch_migrations
        self._bootstrap_mode = bool(bootstrap_mode)
        self._migration_timeout = migration_timeout
        self._non_blocking_migrations = (
            non_blocking_migrations
            if non_blocking_migrations is not None
            else self._bootstrap_mode
        )
        if run_migrations:
            self.apply_migrations(
                apply_nonessential=apply_nonessential_migrations,
                batch=batch_migrations,
                timeout=migration_timeout,
                non_blocking=self._non_blocking_migrations,
            )
        meta_path = Path(vector_index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    def apply_migrations(
        self,
        *,
        apply_nonessential: bool | None = None,
        batch: bool | None = None,
        timeout: float | None = None,
        non_blocking: bool | None = None,
    ) -> None:
        """Ensure the database schema exists using a single transaction."""

        if self._schema_initialised:
            return

        non_blocking = (
            self._non_blocking_migrations
            if non_blocking is None
            else non_blocking
        )
        timeout = self._migration_timeout if timeout is None else timeout
        if non_blocking and timeout is None:
            timeout = 0.0
        if non_blocking and self._schema_ready_cached:
            logger.info(
                "InfoDB skipping migrations using cached schema readiness (non_blocking=%s)",
                non_blocking,
            )
            return

        apply_nonessential = (
            self._apply_nonessential_migrations
            if apply_nonessential is None
            else apply_nonessential
        )
        batch = self._batch_migrations if batch is None else batch

        conn = self.conn
        previous_bootstrap_safe = getattr(conn, "audit_bootstrap_safe", False)
        conn.audit_bootstrap_safe = True
        prior_busy_timeout: int | None = None

        def _restore_busy_timeout() -> None:
            if prior_busy_timeout is None:
                return
            try:
                conn.execute(f"PRAGMA busy_timeout={prior_busy_timeout}")
            except Exception:  # pragma: no cover - best effort restore
                logger.debug("failed to restore busy_timeout", exc_info=True)

        if timeout is not None:
            try:
                prior_busy_timeout = int(
                    conn.execute("PRAGMA busy_timeout").fetchone()[0]
                )
                conn.execute(f"PRAGMA busy_timeout={int(max(timeout, 0) * 1000)}")
            except Exception:  # pragma: no cover - busy timeout best effort
                prior_busy_timeout = None

        start = time.perf_counter()
        migration_success = False
        self.has_fts = False
        try:
            if batch:
                conn.execute("BEGIN")
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS info(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id INTEGER,
                        contrarian_id INTEGER,
                        title TEXT,
                        summary TEXT,
                        tags TEXT,
                        category TEXT,
                        type TEXT,
                        content TEXT,
                        data_depth REAL,
                        source_url TEXT,
                        notes TEXT,
                        associated_bots TEXT,
                        associated_errors TEXT,
                        performance_data TEXT,
                        timestamp REAL,
                        energy INTEGER,
                        corroboration_count INTEGER
                    )
                    """
                )
                cols = {
                    r[1]
                    for r in conn.execute("PRAGMA table_info(info)").fetchall()
                }
                for name, stmt in {
                    "model_id": "ALTER TABLE info ADD COLUMN model_id INTEGER",
                    "contrarian_id": "ALTER TABLE info ADD COLUMN contrarian_id INTEGER",
                    "summary": "ALTER TABLE info ADD COLUMN summary TEXT",
                    "data_depth": "ALTER TABLE info ADD COLUMN data_depth REAL",
                    "source_url": "ALTER TABLE info ADD COLUMN source_url TEXT",
                    "notes": "ALTER TABLE info ADD COLUMN notes TEXT",
                    "energy": "ALTER TABLE info ADD COLUMN energy INTEGER",
                    "corroboration_count": "ALTER TABLE info ADD COLUMN corroboration_count INTEGER",
                }.items():
                    if name not in cols:
                        conn.execute(stmt)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS info_workflows(info_id INTEGER, workflow_id INTEGER)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS info_enhancements(info_id INTEGER, enhancement_id INTEGER)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS info_bots(info_id INTEGER, bot_id INTEGER)"
                )
                if apply_nonessential:
                    try:
                        conn.execute(
                            "CREATE VIRTUAL TABLE IF NOT EXISTS info_fts USING fts5(title, tags, content)"
                        )
                        conn.execute(
                            "INSERT OR IGNORE INTO info_fts(rowid, title, tags, content) SELECT id, title, tags, content FROM info"
                        )
                        self.has_fts = True
                    except sqlite3.OperationalError:
                        self.has_fts = False
                conn.commit()
                migration_success = True
            except Exception:
                if batch:
                    conn.rollback()
                raise
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                logger.info(
                    "InfoDB migrations deferred due to database lock (non_blocking=%s)",
                    non_blocking,
                )
                return
            raise
        finally:
            _restore_busy_timeout()
            conn.audit_bootstrap_safe = previous_bootstrap_safe
            if migration_success:
                self._schema_initialised = True
                self._schema_ready_cached = True
                elapsed = time.perf_counter() - start
                logger.info(
                    "InfoDB migrations completed in %.3fs (batch=%s, nonessential=%s, bootstrap_fast_path=%s)",
                    elapsed,
                    batch,
                    apply_nonessential,
                    non_blocking,
                )

    def set_current_model(self, model_id: int) -> None:
        """Persist the current model id for future inserts."""
        self.current_model_id = model_id

    def set_current_contrarian(self, contrarian_id: int) -> None:
        """Persist the current contrarian id for future inserts."""
        self.current_contrarian_id = contrarian_id

    @auto_link({"workflows": "link_workflow", "enhancements": "link_enhancement"})
    def add(
        self,
        item: ResearchItem,
        embedding: Optional[List[float]] = None,
        *,
        workflows: Iterable[int] | None = None,
        enhancements: Iterable[int] | None = None,
    ) -> int:
        if not item.model_id:
            item.model_id = self.current_model_id
        if not item.contrarian_id:
            item.contrarian_id = self.current_contrarian_id
        if not item.summary:
            item.summary = summarise_text(item.content, ratio=0.2)
        if not item.data_depth:
            item.data_depth = min(len(item.content) / 500.0, 1.0)
        if not item.timestamp:
            item.timestamp = time.time()
        if not item.type_:
            item.type_ = "text"
        tags = ",".join(item.tags)
        bots = ",".join(item.associated_bots)
        errors = ",".join(item.associated_errors)

        # cross reference existing items based on energy level
        notes = []
        corroboration_count = 0
        if item.energy > 1:
            matches = self.search(item.topic)
            if item.energy >= 3:
                matches.extend(self.items_for_model(item.model_id))
            unique: dict[int, ResearchItem] = {m.item_id: m for m in matches}
            corroboration_count = len(unique)
            for m in unique.values():
                if item.energy >= 3 or m.model_id == item.model_id:
                    reasons = []
                    if item.topic.lower() in m.topic.lower() or m.topic.lower() in item.topic.lower():
                        reasons.append("topic")
                    if m.model_id == item.model_id:
                        reasons.append("model")
                    notes.append(f"xref {m.item_id} ({', '.join(reasons)})")
            if item.energy >= 3:
                if self.text_bot:
                    try:
                        self.text_bot.process([item.topic], [], ratio=0.2)
                        notes.append("validated:text")
                        corroboration_count += 1
                    except Exception:
                        notes.append("validated:text_fail")
                if self.video_bot:
                    try:
                        self.video_bot.process(item.topic, ratio=0.2)
                        notes.append("validated:video")
                        corroboration_count += 1
                    except Exception:
                        notes.append("validated:video_fail")
        item.notes = "; ".join(notes)
        item.corroboration_count = corroboration_count

        # embedding will be generated via add_embedding
        item.embedding = embedding if embedding is not None else None

        conn = self.router.get_connection("information")
        cur = conn.execute(
            """
            INSERT INTO info(
                model_id, contrarian_id, title, summary, tags, category, type, content,
                data_depth, source_url, notes, associated_bots, associated_errors,
                performance_data, timestamp, energy, corroboration_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
            (
                item.model_id,
                item.contrarian_id,
                item.title or item.topic,
                item.summary,
                tags,
                item.category,
                item.type_,
                item.content,
                item.data_depth,
                item.source_url,
                item.notes,
                bots,
                errors,
                item.performance_data,
                item.timestamp,
                item.energy,
                item.corroboration_count,
            ),
        )
        item.item_id = cur.lastrowid
        if self.has_fts:
            try:
                conn.execute(
                    "INSERT INTO info_fts(rowid, title, tags, content) VALUES (?,?,?,?)",
                    (
                        item.item_id,
                        item.title or item.topic,
                        tags,
                        item.content,
                    ),
                )
            except sqlite3.OperationalError:
                self.has_fts = False
        conn.commit()
        item.embedding = None
        try:
            self.add_embedding(
                item.item_id,
                item,
                "info",
                source_id=str(item.item_id),
            )
            item.embedding = getattr(self, "_metadata", {}).get(str(item.item_id), {}).get("vector")
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception(
                "embedding hook failed for %s: %s", item.item_id, exc
            )

        if self.event_bus:
            try:
                self.event_bus.publish("info:new", dataclasses.asdict(item))
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception as exc:
                logging.getLogger(__name__).error("publish failed: %s", exc)
        if self.menace_db:
            try:
                self._insert_menace(item, workflows or [])
            except Exception as exc:
                warnings.warn(f"MenaceDB insert failed: {exc}")
        return int(item.item_id)

    def update(self, info_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        params = list(fields.values()) + [info_id]
        conn = self.router.get_connection("information")
        conn.row_factory = sqlite3.Row
        conn.execute(f"UPDATE info SET {sets} WHERE id=?", params)
        row = conn.execute("SELECT * FROM info WHERE id=?", (info_id,)).fetchone()
        conn.commit()
        if not row:
            return
        item = ResearchItem(
            topic=row["title"],
            content=row["content"],
            timestamp=row["timestamp"],
            title=row["title"],
            tags=row["tags"].split(",") if row["tags"] else [],
            item_id=row["id"],
            category=row["category"] or "",
            type_=row["type"] or "",
            associated_bots=row["associated_bots"].split(",") if row["associated_bots"] else [],
            associated_errors=row["associated_errors"].split(",") if row["associated_errors"] else [],
            performance_data=row["performance_data"] or "",
            summary=row["summary"] or "",
            model_id=row["model_id"] or 0,
            contrarian_id=row["contrarian_id"] or 0,
            data_depth=row["data_depth"] or 0.0,
            source_url=row["source_url"] or "",
            notes=row["notes"] or "",
            energy=row["energy"] or 1,
            corroboration_count=row["corroboration_count"] or 0,
        )
        try:
            self.add_embedding(
                info_id,
                item,
                "info",
                source_id=str(info_id),
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s: %s", info_id, exc)
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception as exc:
                logging.getLogger(__name__).error("publish failed: %s", exc)

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(self) -> Iterator[tuple[int, sqlite3.Row, str]]:
        """Yield research info rows for embedding backfill."""
        cur = self.conn.execute("SELECT * FROM info")
        for row in cur.fetchall():
            yield row["id"], row, "info"

    def license_text(self, rec: Any) -> str | None:
        if isinstance(rec, (ResearchItem, dict, sqlite3.Row)):
            return self._embed_text(rec)
        return None

    def log_license_violation(self, path: str, license_name: str, hash_: str) -> None:
        try:  # pragma: no cover - best effort
            CodeDB = importlib.import_module("code_database").CodeDB
            CodeDB().log_license_violation(path, license_name, hash_)
        except Exception:
            pass

    def _flatten_fields(self, data: dict[str, Any]) -> list[str]:
        pairs: list[str] = []

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

    def _embed_text(self, rec: ResearchItem | dict[str, Any] | sqlite3.Row) -> str:
        """Return a flattened ``key=value`` string for ``rec``.

        All fields of :class:`ResearchItem` are first converted to a mapping
        using :func:`dataclasses.asdict`.  For plain ``dict`` or ``sqlite3.Row``
        instances the data is copied so that comma separated list fields can be
        normalised back into lists.  The resulting mapping is then flattened
        into ``key=value`` pairs, including any nested data, before being
        joined into a single space separated string.
        """

        if isinstance(rec, ResearchItem):
            data: dict[str, Any] = dataclasses.asdict(rec)
        elif isinstance(rec, sqlite3.Row):
            data = dict(rec)
        else:
            data = dict(rec)

        # normalise common comma separated fields back into lists
        for key in ("tags", "associated_bots", "associated_errors", "categories"):
            val = data.get(key)
            if isinstance(val, str):
                data[key] = [v for v in val.split(",") if v]

        return " ".join(self._flatten_fields(data))

    def vector(self, rec: Any) -> list[float] | None:
        """Return an embedding for ``rec`` or a stored record id."""

        if isinstance(rec, int) or (isinstance(rec, str) and rec.isdigit()):
            return getattr(self, "_metadata", {}).get(str(rec), {}).get("vector")

        text = self._embed_text(rec)
        return self._embed(text) if text else None

    def _embed(self, text: str) -> list[float]:
        """Encode ``text`` to a vector (overridable for tests)."""
        return self.encode_text(text)

    def search_by_vector(
        self, vector: Iterable[float], top_k: int = 5
    ) -> List[ResearchItem]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[ResearchItem] = []
        for rec_id, dist in matches:
            row = self.conn.execute(
                "SELECT * FROM info WHERE id=?", (rec_id,)
            ).fetchone()
            if row:
                item = ResearchItem(
                    topic=row["title"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    title=row["title"],
                    tags=row["tags"].split(",") if row["tags"] else [],
                    item_id=row["id"],
                    category=row["category"] or "",
                    type_=row["type"] or "",
                    associated_bots=row["associated_bots"].split(",") if row["associated_bots"] else [],
                    associated_errors=row["associated_errors"].split(",") if row["associated_errors"] else [],
                    performance_data=row["performance_data"] or "",
                    summary=row["summary"] or "",
                    model_id=row["model_id"] or 0,
                    contrarian_id=row["contrarian_id"] or 0,
                    data_depth=row["data_depth"] or 0.0,
                    source_url=row["source_url"] or "",
                    notes=row["notes"] or "",
                    energy=row["energy"] or 1,
                    corroboration_count=row["corroboration_count"] or 0,
                    embedding=getattr(self, "_metadata", {}).get(str(row["id"]), {}).get("vector"),
                )
                setattr(item, "_distance", dist)
                results.append(item)
        return results

    def search(self, term: str) -> List[ResearchItem]:
        pattern = f"%{term.lower()}%"
        conn = self.router.get_connection("information")
        conn.row_factory = sqlite3.Row
        if self.has_fts:
            try:
                rows = conn.execute(
                    """
                    SELECT i.id, i.model_id, i.contrarian_id, i.title, i.summary, i.tags, i.category, i.type, i.content,
                           i.data_depth, i.source_url, i.notes, i.associated_bots, i.associated_errors,
                           i.performance_data, i.timestamp, i.energy, i.corroboration_count
                    FROM info AS i JOIN info_fts f ON f.rowid = i.id
                    WHERE info_fts MATCH ?
                    """,
                    (f"{term}*",),
                ).fetchall()
            except sqlite3.OperationalError:
                self.has_fts = False
                rows = conn.execute(
                    """
                    SELECT id, model_id, contrarian_id, title, summary, tags, category, type, content,
                           data_depth, source_url, notes, associated_bots, associated_errors,
                           performance_data, timestamp, energy, corroboration_count
                    FROM info
                    WHERE LOWER(title) LIKE ? OR LOWER(tags) LIKE ?
                    """,
                    (pattern, pattern),
                ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, model_id, contrarian_id, title, summary, tags, category, type, content,
                       data_depth, source_url, notes, associated_bots, associated_errors,
                       performance_data, timestamp, energy, corroboration_count
                FROM info
                WHERE LOWER(title) LIKE ? OR LOWER(tags) LIKE ?
                """,
                (pattern, pattern),
            ).fetchall()
        results: List[ResearchItem] = []
        for row in rows:
            results.append(
                ResearchItem(
                    topic=row["title"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    title=row["title"],
                    tags=row["tags"].split(",") if row["tags"] else [],
                    item_id=row["id"],
                    category=row["category"] or "",
                    type_=row["type"] or "",
                    associated_bots=row["associated_bots"].split(",") if row["associated_bots"] else [],
                    associated_errors=row["associated_errors"].split(",") if row["associated_errors"] else [],
                    performance_data=row["performance_data"] or "",
                    summary=row["summary"] or "",
                    model_id=row["model_id"] or 0,
                    contrarian_id=row["contrarian_id"] or 0,
                    data_depth=row["data_depth"] or 0.0,
                    source_url=row["source_url"] or "",
                    notes=row["notes"] or "",
                    energy=row["energy"] or 1,
                    corroboration_count=row["corroboration_count"] or 0,
                    embedding=self.vector(row["id"]),
                )
            )
        return results

    # ------------------------------------------------------------------
    def link_workflow(self, info_id: int, workflow_id: int) -> None:
        conn = self.router.get_connection("information")
        conn.execute(
            "INSERT INTO info_workflows(info_id, workflow_id) VALUES (?, ?)",
            (info_id, workflow_id),
        )
        conn.commit()

    def workflows_for(self, info_id: int) -> List[int]:
        conn = self.router.get_connection("information")
        rows = conn.execute(
            "SELECT workflow_id FROM info_workflows WHERE info_id=?",
            (info_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def link_enhancement(self, info_id: int, enhancement_id: int) -> None:
        conn = self.router.get_connection("information")
        conn.execute(
            "INSERT INTO info_enhancements(info_id, enhancement_id) VALUES (?, ?)",
            (info_id, enhancement_id),
        )
        conn.commit()

    def link_bot(self, info_id: int, bot_id: int) -> None:
        conn = self.router.get_connection("information")
        conn.execute(
            "INSERT INTO info_bots(info_id, bot_id) VALUES (?, ?)",
            (info_id, bot_id),
        )
        conn.commit()

    def enhancements_for(self, info_id: int) -> List[int]:
        conn = self.router.get_connection("information")
        rows = conn.execute(
            "SELECT enhancement_id FROM info_enhancements WHERE info_id=?",
            (info_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def bots_for(self, info_id: int) -> List[int]:
        conn = self.router.get_connection("information")
        rows = conn.execute(
            "SELECT bot_id FROM info_bots WHERE info_id=?",
            (info_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def items_for_model(self, model_id: int) -> List[ResearchItem]:
        conn = self.router.get_connection("information")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM info WHERE model_id=?",
            (model_id,),
        ).fetchall()
        items: List[ResearchItem] = []
        for row in rows:
            items.append(
                ResearchItem(
                    topic=row["title"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    title=row["title"],
                    tags=row["tags"].split(",") if row["tags"] else [],
                    item_id=row["id"],
                    category=row["category"] or "",
                    type_=row["type"] or "",
                    associated_bots=row["associated_bots"].split(",") if row["associated_bots"] else [],
                    associated_errors=row["associated_errors"].split(",") if row["associated_errors"] else [],
                    performance_data=row["performance_data"] or "",
                    summary=row["summary"] or "",
                    model_id=row["model_id"] or 0,
                    contrarian_id=row["contrarian_id"] or 0,
                    data_depth=row["data_depth"] or 0.0,
                    source_url=row["source_url"] or "",
                    notes=row["notes"] or "",
                    energy=row["energy"] or 1,
                    corroboration_count=row["corroboration_count"] or 0,
                    embedding=self.vector(row["id"]),
                )
            )
        return items

    # ------------------------------------------------------------------
    def _insert_menace(self, item: ResearchItem, workflows: Iterable[int]) -> None:
        """Insert item into the Menace database if configured."""
        if not self.menace_db:
            return
        mdb = self.menace_db
        tags = ",".join(item.tags)
        with mdb.engine.begin() as conn:
            res = conn.execute(
                mdb.information.insert().values(
                    data_type=item.type_,
                    source_url=item.source_url,
                    date_collected=datetime.utcnow().isoformat(),
                    summary=item.summary,
                    validated=True,
                    validation_notes=item.notes,
                    keywords=tags,
                    data_depth_score=item.data_depth,
                )
            )
            info_id = int(res.inserted_primary_key[0])
            if item.embedding:
                conn.execute(
                    mdb.information_embeddings.insert().prefix_with("OR REPLACE").values(
                        record_id=str(info_id),
                        vector=json.dumps(item.embedding),
                        created_at=datetime.utcnow().isoformat(),
                        embedding_version=self.embedding_version,
                        kind="info",
                        source_id=str(info_id),
                    )
                )
            if item.model_id:
                row = conn.execute(
                    mdb.models.select().where(mdb.models.c.model_id == item.model_id)
                ).fetchone()
                if row:
                    conn.execute(
                        mdb.information_models.insert().values(
                            info_id=info_id, model_id=item.model_id
                        )
                    )
                else:
                    warnings.warn(f"invalid model_id {item.model_id}")
            for wid in workflows:
                row = conn.execute(
                    mdb.workflows.select().where(mdb.workflows.c.workflow_id == wid)
                ).fetchone()
                if row:
                    conn.execute(
                        mdb.information_workflows.insert().values(
                            info_id=info_id, workflow_id=wid
                        )
                    )
                else:
                    warnings.warn(f"invalid workflow_id {wid}")
            for bname in item.associated_bots:
                b_row = conn.execute(
                    mdb.bots.select().where(mdb.bots.c.bot_name == bname)
                ).fetchone()
                if b_row and hasattr(mdb, "information_bots"):
                    conn.execute(
                        mdb.information_bots.insert().values(
                            info_id=info_id, bot_id=int(b_row["bot_id"])
                        )
                    )

    def migrate_to_menace(self) -> None:
        """Insert all stored items into the configured MenaceDB."""
        if not self.menace_db:
            return
        for item in self.search(""):
            try:
                wfs = self.workflows_for(item.item_id)
                self._insert_menace(item, wfs)
            except Exception as exc:
                logger.exception(
                    "Failed to migrate item %s to MenaceDB: %s", item, exc
                )


__all__ = ["ResearchItem", "InfoDB"]
