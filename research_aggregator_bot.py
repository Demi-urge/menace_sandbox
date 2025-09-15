"""Research Aggregator Bot for autonomous multi-layered research."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds

from .coding_bot_interface import self_coding_managed
from .self_coding_manager import SelfCodingManager, internalize_coding_bot
from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
from vector_service.context_builder import ContextBuilder
from typing import TYPE_CHECKING
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator
import sqlite3
import time
from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
from typing import Any, Iterable, List, Optional, Iterator
import importlib

from .auto_link import auto_link

from .unified_event_bus import UnifiedEventBus
import json
import os
import logging
import warnings
from datetime import datetime

from .chatgpt_enhancement_bot import (
    EnhancementDB,
    ChatGPTEnhancementBot,
    Enhancement,
)
from .chatgpt_prediction_bot import ChatGPTPredictionBot, IdeaFeatures
from .text_research_bot import TextResearchBot
try:
    from .video_research_bot import VideoResearchBot
except Exception:  # pragma: no cover - optional dependency
    VideoResearchBot = None  # type: ignore
from .chatgpt_research_bot import ChatGPTResearchBot, Exchange, summarise_text
from .database_manager import get_connection, DB_PATH
from .capital_management_bot import CapitalManagementBot
from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from vector_service import EmbeddableDBMixin, ContextBuilder
from snippet_compressor import compress_snippets
try:
    from .menace_db import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = create_context_builder()
engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_context_builder)
pipeline = ModelAutomationPipeline(context_builder=_context_builder)
evolution_orchestrator = get_orchestrator("ResearchAggregatorBot", data_bot, engine)
_th = get_thresholds("ResearchAggregatorBot")
persist_sc_thresholds(
    "ResearchAggregatorBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
    test_failure_increase=_th.test_failure_increase,
)
manager = internalize_coding_bot(
    "ResearchAggregatorBot",
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    evolution_orchestrator=evolution_orchestrator,
    threshold_service=ThresholdService(),
    roi_threshold=_th.roi_drop,
    error_threshold=_th.error_increase,
    test_failure_threshold=_th.test_failure_increase,
)

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


class ResearchMemory:
    """Tiered memory with decay for storing research items."""

    def __init__(self) -> None:
        self.short: List[ResearchItem] = []
        self.medium: List[ResearchItem] = []
        self.long: List[ResearchItem] = []

    def add(self, item: ResearchItem, layer: str = "short") -> None:
        getattr(self, layer).append(item)

    def search(self, topic: str) -> List[ResearchItem]:
        topic = topic.lower()
        results: List[ResearchItem] = []
        for layer in [self.short, self.medium, self.long]:
            results.extend([i for i in layer if topic in i.topic.lower()])
        return results

    def decay(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.short = [i for i in self.short if now - i.timestamp < 60]
        self.medium = [i for i in self.medium if now - i.timestamp < 3600]

    def archive(self, items: Iterable[ResearchItem]) -> None:
        for it in items:
            if it not in self.long:
                self.long.append(it)


class InfoDB(EmbeddableDBMixin):
    """SQLite storage for collected research."""

    DB_FILE = "information.db"

    def __init__(
        self,
        path: Path = Path("information.db"),
        current_model_id: int = 0,
        current_contrarian_id: int = 0,
        text_bot: Optional[TextResearchBot] = None,
        video_bot: Optional[VideoResearchBot] = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        menace_db: "MenaceDB" | None = None,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "information_embeddings.index",
        embedding_version: int = 1,
        router: DBRouter | None = None,
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
        conn = self.conn
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
        cols = [r[1] for r in conn.execute("PRAGMA table_info(info)").fetchall()]
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
        meta_path = Path(vector_index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
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


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class ResearchAggregatorBot:
    """Collects, refines and stores research with energy-based depth."""

    def __init__(
        self,
        requirements: Iterable[str],
        memory: Optional[ResearchMemory] = None,
        info_db: Optional[InfoDB] = None,
        enhancements_db: Optional[EnhancementDB] = None,
        enhancement_bot: Optional[ChatGPTEnhancementBot] = None,
        prediction_bot: Optional[ChatGPTPredictionBot] = None,
        text_bot: Optional[TextResearchBot] = None,
        video_bot: Optional[VideoResearchBot] = None,
        chatgpt_bot: Optional[ChatGPTResearchBot] = None,
        capital_manager: Optional[CapitalManagementBot] = None,
        db_router: Optional[DBRouter] = None,
        enhancement_interval: float = 300.0,
        cache_ttl: float = 3600.0,
        *,
        manager: SelfCodingManager | None = None,
        context_builder: ContextBuilder,
    ) -> None:
        builder = context_builder
        if builder is None:
            raise ValueError("ContextBuilder is required")
        mgr = manager or globals().get("manager")
        self.manager = mgr
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = data_bot
        self.requirements = list(requirements)
        self.memory = memory or ResearchMemory()
        self.info_db = info_db or InfoDB()
        self.db_router = db_router or DBRouter(info_db=self.info_db)
        self.enh_db = enhancements_db or EnhancementDB()
        self.enhancement_bot = enhancement_bot
        self.prediction_bot = prediction_bot
        self.text_bot = text_bot
        self.video_bot = video_bot
        self.chatgpt_bot = chatgpt_bot
        if self.chatgpt_bot and getattr(self.chatgpt_bot, "send_callback", None) is None:
            self.chatgpt_bot.send_callback = self.receive_chatgpt
        self.capital_manager = capital_manager or CapitalManagementBot()
        self.enhancement_interval = enhancement_interval
        self._last_enhancement_time = 0.0
        self.sources_queried: List[str] = []
        self.cache_ttl = cache_ttl
        self.cache: dict[str, tuple[float, List[ResearchItem]]] = {}
        try:
            builder.refresh_db_weights()
        except Exception:
            logger.exception("Failed to initialise ContextBuilder")
            raise
        self.context_builder = builder

    # ------------------------------------------------------------------
    def _increment_enh_count(self, model_id: int) -> None:
        """Increment enhancement counter in models.db if possible."""
        try:
            with get_connection(DB_PATH) as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
                if "enhancement_count" not in cols:
                    conn.execute("ALTER TABLE models ADD COLUMN enhancement_count INTEGER DEFAULT 0")
                cur = conn.execute(
                    "SELECT enhancement_count FROM models WHERE id=?",
                    (model_id,),
                )
                row = cur.fetchone()
                count = int(row[0]) if row and row[0] is not None else 0
                conn.execute(
                    "UPDATE models SET enhancement_count=? WHERE id=?",
                    (count + 1, model_id),
                )
        except Exception as exc:
            logger.exception(
                "Failed to increment enhancement count for model %s: %s",
                model_id,
                exc,
            )

    def _info_ratio(self, energy: int) -> int:
        try:
            ratio = self.capital_manager.info_ratio(float(energy))
        except Exception:
            ratio = float(energy)
        return max(1, int(round(ratio)))

    def _compressed_context(self, query: str) -> str:
        try:
            ctx = str(self.context_builder.build(query))
        except Exception as exc:
            logger.exception("Context build failed for %s: %s", query, exc)
            ctx = ""
        return compress_snippets({"snippet": ctx}).get("snippet", ctx)

    def _query_local(self, topic: str) -> List[ResearchItem]:
        items = []
        lower = topic.lower()
        for it in self.info_db.search(topic):
            if it.title.lower() == lower or lower in [t.lower() for t in it.tags]:
                items.append(it)
        for enh in self.enh_db.fetch():
            text = f"{enh.idea} {enh.rationale}".lower()
            if lower in text:
                items.append(
                    ResearchItem(
                        topic=topic,
                        content=f"{enh.idea}: {enh.rationale}",
                        timestamp=time.time(),
                        categories=["enhancement"],
                    )
                )
        return items

    def _missing_data_types(self, items: Iterable[ResearchItem], topic: str) -> List[str]:
        """Return data type names that are absent for the given topic."""
        have = {it.type_.lower() for it in items if it.topic == topic}
        required = {"text", "video", "chatgpt"}
        return [t for t in required if t not in have]

    def _gather_online(self, topic: str, energy: int, amount: int = 1) -> List[ResearchItem]:
        results: List[ResearchItem] = []
        for i in range(max(1, amount)):
            content = f"web data {i} for {topic} with energy {energy}"
            results.append(
                ResearchItem(
                    topic=topic,
                    content=content,
                    timestamp=time.time(),
                    type_="web",
                )
            )
            try:
                self.db_router.insert_info(results[-1])
            except Exception as exc:
                logger.exception("Failed to insert gathered online data: %s", exc)
        return results

    def _delegate_sub_bots(
        self,
        topic: str,
        energy: int,
        amount: int = 1,
        missing: Optional[Iterable[str]] = None,
    ) -> List[ResearchItem]:
        results: List[ResearchItem] = []
        collected_text = []
        queried: List[str] = []
        targets = set(missing or ["text", "video", "chatgpt"])
        ctx = self._compressed_context(topic)
        for _ in range(max(1, amount)):
            if "text" in targets and self.text_bot:
                try:
                    texts = self.text_bot.process([topic], [], ratio=0.2)
                    queried.append("text")
                    for t in texts:
                        item = ResearchItem(
                            topic=topic,
                            content=t.content,
                            timestamp=time.time(),
                            source_url=t.url,
                            type_="text",
                            associated_bots=[self.text_bot.__class__.__name__],
                        )
                        results.append(item)
                        try:
                            self.db_router.insert_info(item)
                        except Exception as exc:
                            logger.exception("Failed to insert text info: %s", exc)
                        collected_text.append(t.content)
                except Exception as exc:
                    logger.exception("Text bot failed for %s: %s", topic, exc)
            if "video" in targets and self.video_bot:
                try:
                    vids = self.video_bot.process(topic, ratio=0.2)
                    queried.append("video")
                    for v in vids:
                        item = ResearchItem(
                            topic=topic,
                            content=v.summary,
                            timestamp=time.time(),
                            source_url=v.url,
                            type_="video",
                            associated_bots=[self.video_bot.__class__.__name__],
                        )
                        results.append(item)
                        try:
                            self.db_router.insert_info(item)
                        except Exception as exc:
                            logger.exception("Failed to insert video info: %s", exc)
                except Exception as exc:
                    logger.exception("Video bot failed for %s: %s", topic, exc)
            if self.enhancement_bot and (not missing or "enhancement" in targets):
                try:
                    try:
                        enhs = self.enhancement_bot.propose(
                            topic,
                            num_ideas=1,
                            context=ctx,
                            context_builder=self.context_builder,
                        )
                    except TypeError:
                        enhs = self.enhancement_bot.propose(
                            topic, num_ideas=1, context=ctx
                        )
                    for enh in enhs:
                        evaluation = None
                        if self.prediction_bot:
                            try:
                                try:
                                    evaluation = self.prediction_bot.evaluate_enhancement(
                                        enh.idea,
                                        enh.rationale,
                                        context_builder=self.context_builder,
                                    )
                                except TypeError:
                                    evaluation = self.prediction_bot.evaluate_enhancement(
                                        enh.idea, enh.rationale
                                    )
                                enh.score = evaluation.value
                            except Exception:
                                evaluation = None
                                logger.exception("Enhancement prediction failed for %s", topic)
                        self.enh_db.add(enh)
                        item = ResearchItem(
                            topic=topic,
                            content=f"{enh.idea}: {enh.rationale}",
                            timestamp=time.time(),
                            categories=["enhancement"],
                            summary=evaluation.description if evaluation else "",
                            notes=evaluation.reason if evaluation else "",
                            quality=evaluation.value if evaluation else 0.0,
                            type_="enhancement",
                            associated_bots=[self.enhancement_bot.__class__.__name__],
                        )
                        results.append(item)
                        try:
                            self.db_router.insert_info(item)
                        except Exception as exc:
                            logger.exception("Failed to insert enhancement info: %s", exc)
                        self._increment_enh_count(self.info_db.current_model_id)
                except Exception as exc:
                    logger.exception("Enhancement bot failed for %s: %s", topic, exc)
        if "chatgpt" in targets and self.chatgpt_bot:
            try:
                instruction = topic
                if collected_text:
                    joined = " ".join(collected_text)
                    instruction = f"Summarise the following about {topic}: {joined}"
                if ctx:
                    instruction = f"{ctx}\n\n{instruction}"
                try:
                    res = self.chatgpt_bot.process(
                        instruction,
                        depth=1,
                        ratio=0.2,
                        context_builder=self.context_builder,
                    )
                except TypeError:
                    res = self.chatgpt_bot.process(instruction, depth=1, ratio=0.2)
                item = ResearchItem(
                    topic=topic,
                    content=res.summary,
                    timestamp=time.time(),
                    type_="chatgpt",
                    associated_bots=[self.chatgpt_bot.__class__.__name__],
                )
                results.append(item)
                queried.append("chatgpt")
                try:
                    self.db_router.insert_info(item)
                except Exception as exc:
                    logger.exception("Failed to insert chatgpt info: %s", exc)
            except Exception as exc:
                logger.exception("ChatGPT bot failed for %s: %s", topic, exc)
        if not results:
            content = f"sub bot research for {topic}"
            results.append(ResearchItem(topic=topic, content=content, timestamp=time.time()))
        for q in queried:
            if q not in self.sources_queried:
                self.sources_queried.append(q)
        return results

    @staticmethod
    def _refine(items: Iterable[ResearchItem]) -> List[ResearchItem]:
        seen: set[str] = set()
        refined: List[ResearchItem] = []
        for it in items:
            if it.content in seen:
                continue
            seen.add(it.content)
            refined.append(it)
        return refined

    def receive_chatgpt(self, convo: Iterable[Exchange], summary: str) -> None:
        """Store ChatGPT findings in memory."""
        item = ResearchItem(
            topic="chatgpt",
            content=summary,
            timestamp=time.time(),
            categories=["chatgpt"],
        )
        self.memory.add(item, layer="short")

    def _is_complete(self, items: Iterable[ResearchItem]) -> bool:
        topics = {it.topic for it in items}
        return all(req in topics for req in self.requirements)

    def _maybe_enhance(self, topic: str, reason: str) -> None:
        """Request an enhancement from the enhancement bot if available."""
        if self.manager and not self.manager.should_refactor():
            return
        if not self.enhancement_bot:
            return
        ctx = self._compressed_context(topic)
        instruction = f"{reason} about {topic}"
        if ctx:
            instruction = f"{ctx}\n\n{instruction}"
        try:
            try:
                enhancements = self.enhancement_bot.propose(
                    instruction,
                    num_ideas=1,
                    context=ctx or topic,
                    context_builder=self.context_builder,
                )
            except TypeError:
                enhancements = self.enhancement_bot.propose(
                    instruction, num_ideas=1, context=ctx or topic
                )
        except Exception:
            return
        for enh in enhancements:
            evaluation = None
            enh_id = 0
            try:
                if self.prediction_bot:
                    try:
                        evaluation = self.prediction_bot.evaluate_enhancement(
                            enh.idea,
                            enh.rationale,
                            context_builder=self.context_builder,
                        )
                    except TypeError:
                        evaluation = self.prediction_bot.evaluate_enhancement(
                            enh.idea, enh.rationale
                        )
                    enh.score = evaluation.value
                if getattr(self.info_db, "current_model_id", 0):
                    enh.model_ids = [self.info_db.current_model_id]
                bot_id = getattr(self, "bot_id", 0)
                if bot_id:
                    enh.bot_ids = [bot_id]
                workflow_id = getattr(self, "workflow_id", 0)
                if workflow_id:
                    enh.workflow_ids = [workflow_id]
                enh.triggered_by = self.__class__.__name__
                enh_id = self.enh_db.add(enh)
                for mid in enh.model_ids:
                    try:
                        self.enh_db.link_model(enh_id, mid)
                    except Exception as exc:
                        logger.exception("Failed to link model %s to enhancement %s: %s", mid, enh_id, exc)
                for bid in enh.bot_ids:
                    try:
                        self.enh_db.link_bot(enh_id, bid)
                    except Exception as exc:
                        logger.exception("Failed to link bot %s to enhancement %s: %s", bid, enh_id, exc)
                for wid in enh.workflow_ids:
                    try:
                        self.enh_db.link_workflow(enh_id, wid)
                    except Exception as exc:
                        logger.exception(
                            "Failed to link workflow %s to enhancement %s: %s",
                            wid,
                            enh_id,
                            exc,
                        )
                self._increment_enh_count(self.info_db.current_model_id)
            except Exception:
                evaluation = None
            item = ResearchItem(
                topic=topic,
                content=f"{enh.idea}: {enh.rationale}",
                timestamp=time.time(),
                categories=["enhancement"],
                summary=evaluation.description if evaluation else "",
                notes=evaluation.reason if evaluation else "",
                quality=evaluation.value if evaluation else 0.0,
                type_="enhancement",
            )
            self.memory.add(item, layer="medium")
            info_id = 0
            try:
                self.db_router.insert_info(item)
                info_id = item.item_id
            except Exception as exc:
                logger.exception("Failed to insert enhancement info: %s", exc)
            if info_id:
                try:
                    self.info_db.link_enhancement(info_id, enh_id)
                except Exception as exc:
                    logger.exception(
                        "Failed to link enhancement %s to info %s: %s",
                        enh_id,
                        info_id,
                        exc,
                    )

            # gather further research on the enhancement idea
            if self.text_bot:
                try:
                    texts = self.text_bot.process([enh.idea], [], ratio=0.2)
                    for t in texts:
                        r = ResearchItem(
                            topic=enh.idea,
                            content=t.content,
                            timestamp=time.time(),
                            source_url=t.url,
                            type_="text",
                        )
                        self.memory.add(r, layer="short")
                        try:
                            self.db_router.insert_info(r)
                            self.info_db.link_enhancement(r.item_id, enh_id)
                        except Exception as exc:
                            logger.exception(
                                "Failed to store text enhancement research: %s",
                                exc,
                            )
                except Exception:
                    logger.exception("Text bot failed while enhancing %s", enh.idea)
            if self.video_bot:
                try:
                    vids = self.video_bot.process(enh.idea, ratio=0.2)
                    for v in vids:
                        r = ResearchItem(
                            topic=enh.idea,
                            content=v.summary,
                            timestamp=time.time(),
                            source_url=v.url,
                            type_="video",
                        )
                        self.memory.add(r, layer="short")
                        try:
                            self.db_router.insert_info(r)
                            self.info_db.link_enhancement(r.item_id, enh_id)
                        except Exception as exc:
                            logger.exception(
                                "Failed to store video enhancement research: %s",
                                exc,
                            )
                except Exception:
                    logger.exception("Video bot failed while enhancing %s", enh.idea)
            if self.chatgpt_bot:
                try:
                    ctx_enh = self._compressed_context(enh.idea)
                    instruction = enh.idea
                    if ctx_enh:
                        instruction = f"{ctx_enh}\n\n{enh.idea}"
                    try:
                        res = self.chatgpt_bot.process(
                            instruction,
                            depth=1,
                            ratio=0.2,
                            context_builder=self.context_builder,
                        )
                    except TypeError:
                        res = self.chatgpt_bot.process(instruction, depth=1, ratio=0.2)
                    r = ResearchItem(
                        topic=enh.idea,
                        content=res.summary,
                        timestamp=time.time(),
                        type_="chatgpt",
                    )
                    self.memory.add(r, layer="short")
                    try:
                        self.db_router.insert_info(r)
                        self.info_db.link_enhancement(r.item_id, enh_id)
                    except Exception as exc:
                        logger.exception(
                            "Failed to store chatgpt enhancement research: %s",
                            exc,
                        )
                except Exception as exc:
                    logger.exception("ChatGPT bot failed while enhancing %s: %s", enh.idea, exc)

    def _collect_topic(self, topic: str, energy: int) -> List[ResearchItem]:
        if topic in self.sources_queried:
            return []
        self.sources_queried.append(topic)

        existing = self.memory.search(topic)
        local = self._query_local(topic)
        data = existing + local
        if not data:
            amount = self._info_ratio(energy)
            data = self._gather_online(topic, energy, amount)
            if energy > 2:
                data += self._delegate_sub_bots(topic, energy, amount)
        if not data:
            self._maybe_enhance(topic, "Gap detected")
        refined = self._refine(data)
        for it in refined:
            if it in local and ("workflow" in [t.lower() for t in it.tags] or it.category.lower() == "workflow"):
                tag_set = {t.lower() for t in it.tags}
                complete = set(r.lower() for r in self.requirements).issubset(tag_set)
                depth_ok = it.data_depth >= 0.5
                if not complete or not depth_ok:
                    if "partial_reusable" not in tag_set:
                        it.tags.append("partial_reusable")
                elif "reusable" not in tag_set:
                    it.tags.append("reusable")
            self.memory.add(it, layer="medium")
            try:
                self.db_router.insert_info(it)
            except Exception as exc:
                logger.exception("Failed to insert collected topic info: %s", exc)
        return refined

    def process(self, topic: str, energy: int = 1) -> List[ResearchItem]:
        now = time.time()
        if topic in self.cache:
            ts, data = self.cache[topic]
            if now - ts < self.cache_ttl:
                return list(data)

        self.memory.decay()
        now = time.time()
        if now - self._last_enhancement_time >= self.enhancement_interval:
            self._maybe_enhance(topic, "Periodic enhancement")
            self._last_enhancement_time = now
        results = self._collect_topic(topic, energy)
        missing_types = self._missing_data_types(results, topic)
        if energy > 2 and missing_types:
            results.extend(self._delegate_sub_bots(topic, energy, missing=missing_types))

        topics = {it.topic for it in results}
        missing = [req for req in self.requirements if req not in topics]
        attempts = 0
        # continually request missing topics until complete or attempts exceeded
        while missing and attempts < len(self.requirements) * 2:
            for req in list(missing):
                res = self._collect_topic(req, energy)
                results.extend(res)
                miss = self._missing_data_types(res, req)
                if energy > 2 and miss:
                    results.extend(self._delegate_sub_bots(req, energy, missing=miss))
            topics = {it.topic for it in results}
            missing = [req for req in self.requirements if req not in topics]
            attempts += 1
        refined = self._refine(results)
        if self._is_complete(refined):
            self.memory.archive(refined)
            send_to_stage3(refined)
        self.cache[topic] = (time.time(), list(refined))
        return refined


def send_to_stage3(items: Iterable[ResearchItem]) -> None:
    """Forward *items* to Stage 3 via HTTP if ``requests`` is available."""

    try:
        import requests  # type: ignore
        from dataclasses import asdict
    except Exception:  # pragma: no cover - optional dependency
        return

    url = os.getenv("STAGE3_URL")
    if not url:
        return

    payload = [asdict(it) for it in items]
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:  # pragma: no cover - network issues
        logging.getLogger(__name__).warning("Failed to forward items to Stage 3")


__all__ = [
    "ResearchItem",
    "ResearchMemory",
    "InfoDB",
    "EnhancementDB",
    "ChatGPTEnhancementBot",
    "ChatGPTPredictionBot",
    "TextResearchBot",
    "VideoResearchBot",
    "ChatGPTResearchBot",
    "ResearchAggregatorBot",
    "send_to_stage3",
]