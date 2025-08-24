from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, TYPE_CHECKING
from time import time

from .auto_link import auto_link

from .unified_event_bus import UnifiedEventBus
from .retry_utils import publish_with_retry
from vector_service import EmbeddableDBMixin
import warnings
try:
    from .menace_db import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore
    warnings.warn("MenaceDB unavailable, Menace integration disabled.")

from db_router import GLOBAL_ROUTER as router

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .deployment_bot import DeploymentDB


def _serialize_list(items: Iterable[str]) -> str:
    return json.dumps(list(items))


def _deserialize_list(val: str) -> list[str]:
    if not val:
        return []
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return [v for v in val.split(',') if v]


def _safe_json_dumps(data: Any) -> str:
    try:
        result = json.dumps(data)
    except TypeError as exc:
        raise ValueError(f"unserialisable data: {exc}") from exc
    if len(result) > 65535:
        raise ValueError("data blob too large")
    return result


logger = logging.getLogger(__name__)


@dataclass
class BotRecord:
    """Representation of a bot entry."""

    name: str
    type_: str = ""
    tasks: list[str] = field(default_factory=list)
    parent_id: str = ""
    dependencies: list[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    hierarchy_level: str = ""
    purpose: str = ""
    tags: list[str] = field(default_factory=list)
    toolchain: list[str] = field(default_factory=list)
    creation_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modification_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "active"
    version: str = field(default_factory=lambda: os.getenv("BOT_VERSION", ""))
    estimated_profit: float = 0.0
    bid: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"BotRecord(name={self.name!r}, id={self.bid})"

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.name}({self.bid})"


@dataclass
class FailedEvent:
    topic: str
    payload: dict
    attempt: int = 1
    next_retry: float = field(default_factory=time)


@dataclass
class FailedMenace:
    rec: BotRecord
    models: Iterable[int]
    workflows: Iterable[int]
    enhancements: Iterable[int]
    source_menace_id: str
    attempt: int = 1
    next_retry: float = field(default_factory=time)


class BotDB(EmbeddableDBMixin):
    """SQLite database tracking bots and relationships."""

    MAX_RETRIES = 5

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        menace_db: "MenaceDB" | None = None,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "bot_embeddings.index",
        embedding_version: int = 1,
    ) -> None:
        if not router:
            raise RuntimeError("Database router is not initialised")
        self.conn = router.get_connection("bots")
        self.event_bus = event_bus
        self.menace_db = menace_db
        self.failed_events: list[FailedEvent] = []
        self.failed_menace: list[FailedMenace] = []
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                type TEXT,
                tasks TEXT,
                parent_id TEXT,
                dependencies TEXT,
                resources TEXT,
                hierarchy_level TEXT,
                purpose TEXT,
                tags TEXT,
                toolchain TEXT,
                creation_date TEXT,
                last_modification_date TEXT,
                status TEXT,
                version TEXT,
                estimated_profit REAL,
                source_menace_id TEXT DEFAULT ''
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(bots)").fetchall()]
        if "hierarchy_level" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN hierarchy_level TEXT")
        if "purpose" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN purpose TEXT")
        if "tags" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN tags TEXT")
        if "toolchain" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN toolchain TEXT")
        if "version" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN version TEXT")
        if "estimated_profit" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN estimated_profit REAL")
        if "source_menace_id" not in cols:
            self.conn.execute(
                "ALTER TABLE bots ADD COLUMN source_menace_id TEXT DEFAULT ''"
            )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bot_model(bot_id INTEGER, model_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bot_workflow(bot_id INTEGER, workflow_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bot_enhancement(bot_id INTEGER, enhancement_id INTEGER)"
        )
        self.conn.commit()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_bots_name ON bots(name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_bots_type ON bots(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_bots_status ON bots(status)")
        self.conn.commit()
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    # basic helpers -----------------------------------------------------
    def _current_menace_id(self, source_menace_id: str | None) -> str:
        return source_menace_id or (
            router.menace_id if router else os.getenv("MENACE_ID", "")
        )

    def find_by_name(
        self,
        name: str,
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> Optional[Dict[str, Any]]:
        menace_id = self._current_menace_id(source_menace_id)
        query = "SELECT * FROM bots WHERE LOWER(name)=LOWER(?)"
        params: list[Any] = [name]
        if not include_cross_instance:
            query += " AND source_menace_id=?"
            params.append(menace_id)
        cur = self.conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def fetch_all(
        self,
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> list[Dict[str, Any]]:
        menace_id = self._current_menace_id(source_menace_id)
        query = "SELECT * FROM bots"
        params: list[Any] = []
        if not include_cross_instance:
            query += " WHERE source_menace_id=?"
            params.append(menace_id)
        cur = self.conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def by_level(
        self,
        level: str,
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> list[Dict[str, Any]]:
        menace_id = self._current_menace_id(source_menace_id)
        query = "SELECT * FROM bots WHERE hierarchy_level=?"
        params: list[Any] = [level]
        if not include_cross_instance:
            query += " AND source_menace_id=?"
            params.append(menace_id)
        cur = self.conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def deployment_frequency(
        self,
        bot_id: int,
        dep_db: "DeploymentDB",
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> int:
        """Return number of deployment trials recorded for ``bot_id``.

        The ``DeploymentDB`` maintains a ``bot_trials`` table which logs each
        deployment attempt. This helper simply counts rows for the given bot.
        """
        menace_id = self._current_menace_id(source_menace_id)
        query = "SELECT COUNT(*) FROM bot_trials WHERE bot_id=?"
        params: list[Any] = [bot_id]
        if not include_cross_instance:
            query += " AND source_menace_id=?"
            params.append(menace_id)
        cur = dep_db.conn.execute(query, params)
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def retry_failed(self) -> None:
        """Retry any queued events or MenaceDB inserts."""
        now = time()
        if self.event_bus and self.failed_events:
            remaining: list[FailedEvent] = []
            for ev in self.failed_events:
                if ev.next_retry > now:
                    remaining.append(ev)
                    continue
                if not publish_with_retry(self.event_bus, ev.topic, ev.payload):
                    ev.attempt += 1
                    if ev.attempt <= self.MAX_RETRIES:
                        ev.next_retry = now + 2 ** (ev.attempt - 1)
                        remaining.append(ev)
                    else:
                        logger.error("dropping event %s after %s attempts", ev.topic, ev.attempt)
            self.failed_events = remaining
        if self.menace_db and self.failed_menace:
            still_pending: list[FailedMenace] = []
            for item in self.failed_menace:
                if item.next_retry > now:
                    still_pending.append(item)
                    continue
                try:
                    self._insert_menace(
                        item.rec,
                        item.models,
                        item.workflows,
                        item.enhancements,
                        source_menace_id=item.source_menace_id,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.exception("MenaceDB insert retry failed for %s: %s", item.rec.name, exc)
                    item.attempt += 1
                    if item.attempt <= self.MAX_RETRIES:
                        item.next_retry = now + 2 ** (item.attempt - 1)
                        still_pending.append(item)

            self.failed_menace = still_pending

    def _embed_record_on_write(self, bot_id: int, rec: BotRecord | dict[str, Any]) -> None:
        """Best-effort embedding hook for inserts and updates."""

        try:
            self.add_embedding(bot_id, rec, "bot", source_id=str(bot_id))
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s: %s", bot_id, exc)

    def license_text(
        self,
        rec: BotRecord | dict[str, Any],
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> str | None:
        menace_id = self._current_menace_id(source_menace_id)
        if isinstance(rec, (int, str)):
            query = "SELECT purpose, tags, toolchain FROM bots WHERE id=?"
            params: list[Any] = [rec]
            if not include_cross_instance:
                query += " AND source_menace_id=?"
                params.append(menace_id)
            row = self.conn.execute(query, params).fetchone()
            if not row:
                return None
            rec = dict(row)
        if isinstance(rec, BotRecord):
            purpose = rec.purpose
            tags = rec.tags
            toolchain = rec.toolchain
        else:
            purpose = rec.get("purpose", "")
            tags_val = rec.get("tags", [])
            tags = (
                _deserialize_list(tags_val)
                if isinstance(tags_val, str)
                else list(tags_val)
            )
            tc_val = rec.get("toolchain", [])
            toolchain = (
                _deserialize_list(tc_val)
                if isinstance(tc_val, str)
                else list(tc_val)
            )
        return " ".join(filter(None, [purpose, ",".join(tags), ",".join(toolchain)]))

    @auto_link(
        {
            "models": "link_model",
            "workflows": "link_workflow",
            "enhancements": "link_enhancement",
        }
    )
    def add_bot(
        self,
        rec: BotRecord,
        *,
        models: Iterable[int] | None = None,
        workflows: Iterable[int] | None = None,
        enhancements: Iterable[int] | None = None,
        source_menace_id: str | None = None,
    ) -> int:
        menace_id = self._current_menace_id(source_menace_id)
        cur = self.conn.execute(
            """
            INSERT INTO bots(
                name, type, tasks, parent_id, dependencies,
                resources, hierarchy_level, purpose, tags, toolchain,
                creation_date, last_modification_date, status,
                version, estimated_profit, source_menace_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.name,
                rec.type_,
                _serialize_list(rec.tasks),
                rec.parent_id,
                _serialize_list(rec.dependencies),
                _safe_json_dumps(rec.resources),
                rec.hierarchy_level,
                rec.purpose,
                _serialize_list(rec.tags),
                _serialize_list(rec.toolchain),
                rec.creation_date,
                rec.last_modification_date,
                rec.status,
                rec.version,
                rec.estimated_profit,
                menace_id,
            ),
        )
        rec.bid = int(cur.lastrowid)
        self.conn.commit()
        self._embed_record_on_write(rec.bid, rec)
        if self.menace_db:
            try:
                self._insert_menace(
                    rec,
                    models or [],
                    workflows or [],
                    enhancements or [],
                    source_menace_id=source_menace_id
                    or (router.menace_id if router else None),
                )
            except Exception as exc:
                logger.exception("MenaceDB insert failed: %s", exc)
                self.failed_menace.append(
                    FailedMenace(
                        rec,
                        models or [],
                        workflows or [],
                        enhancements or [],
                        router.menace_id if router else os.getenv("MENACE_ID", ""),
                    )
                )
        if self.event_bus:
            payload = dataclasses.asdict(rec)
            if not publish_with_retry(self.event_bus, "bot:new", payload):
                logger.exception("failed to publish bot:new event")
                self.failed_events.append(FailedEvent("bot:new", payload))
            else:
                publish_with_retry(
                    self.event_bus,
                    "embedding:backfill",
                    {"db": self.__class__.__name__},
                )
        return rec.bid

    def update_bot(
        self,
        bot_id: int,
        *,
        include_cross_instance: bool = False,
        source_menace_id: str | None = None,
        **fields: Any,
    ) -> None:
        if not fields:
            return
        if "tags" in fields:
            tags_val = fields["tags"]
            fields["tags"] = (
                _serialize_list(_deserialize_list(tags_val))
                if isinstance(tags_val, str)
                else _serialize_list(tags_val)
            )
        if "toolchain" in fields:
            tc_val = fields["toolchain"]
            fields["toolchain"] = (
                _serialize_list(_deserialize_list(tc_val))
                if isinstance(tc_val, str)
                else _serialize_list(tc_val)
            )
        fields["last_modification_date"] = datetime.utcnow().isoformat()
        sets = ", ".join(f"{k}=?" for k in fields)
        params = list(fields.values()) + [bot_id]
        menace_id = self._current_menace_id(source_menace_id)
        query = f"UPDATE bots SET {sets} WHERE id=?"
        if not include_cross_instance:
            query += " AND source_menace_id=?"
            params.append(menace_id)
        self.conn.execute(query, params)
        self.conn.commit()
        sel_q = "SELECT * FROM bots WHERE id=?"
        sel_params: list[Any] = [bot_id]
        if not include_cross_instance:
            sel_q += " AND source_menace_id=?"
            sel_params.append(menace_id)
        row = self.conn.execute(sel_q, sel_params).fetchone()
        if row:
            self._embed_record_on_write(bot_id, dict(row))
        if self.event_bus:
            payload = {"bot_id": bot_id, **fields}
            if not publish_with_retry(self.event_bus, "bot:update", payload):
                logger.exception("failed to publish bot:update event")
                self.failed_events.append(FailedEvent("bot:update", payload))

    # embedding --------------------------------------------------------
    def iter_records(
        self,
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> Iterable[tuple[Any, Any, str]]:
        menace_id = self._current_menace_id(source_menace_id)
        query = "SELECT * FROM bots"
        params: list[Any] = []
        if not include_cross_instance:
            query += " WHERE source_menace_id=?"
            params.append(menace_id)
        cur = self.conn.execute(query, params)
        for row in cur.fetchall():
            yield row["id"], dict(row), "bot"

    def backfill_embeddings(self) -> None:
        EmbeddableDBMixin.backfill_embeddings(self)

    def vector(
        self,
        rec: Any,
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> list[float]:
        menace_id = self._current_menace_id(source_menace_id)
        if isinstance(rec, (int, str)):
            query = "SELECT purpose, tags, toolchain FROM bots WHERE id=?"
            params: list[Any] = [rec]
            if not include_cross_instance:
                query += " AND source_menace_id=?"
                params.append(menace_id)
            row = self.conn.execute(query, params).fetchone()
            if not row:
                raise ValueError("record not found")
            rec = dict(row)
        if isinstance(rec, BotRecord):
            purpose = rec.purpose
            tags = rec.tags
            toolchain = rec.toolchain
        else:
            purpose = rec.get("purpose", "")
            tags_val = rec.get("tags", [])
            tags = (
                _deserialize_list(tags_val)
                if isinstance(tags_val, str)
                else list(tags_val)
            )
            tc_val = rec.get("toolchain", [])
            toolchain = (
                _deserialize_list(tc_val)
                if isinstance(tc_val, str)
                else list(tc_val)
            )
        text = " ".join(
            filter(None, [purpose, ",".join(tags), ",".join(toolchain)])
        )
        return self.encode_text(text)

    def search_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        *,
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> list[Dict[str, Any]]:
        menace_id = self._current_menace_id(source_menace_id)
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: list[Dict[str, Any]] = []
        for rec_id, dist in matches:
            query = "SELECT * FROM bots WHERE id=?"
            params: list[Any] = [rec_id]
            if not include_cross_instance:
                query += " AND source_menace_id=?"
                params.append(menace_id)
            row = self.conn.execute(query, params).fetchone()
            if row:
                rec = dict(row)
                rec["_distance"] = dist
                results.append(rec)
        return results

    # linking -----------------------------------------------------------
    def link_model(self, bot_id: int, model_id: int) -> None:
        self.conn.execute(
            "INSERT INTO bot_model(bot_id, model_id) VALUES (?, ?)",
            (bot_id, model_id),
        )
        self.conn.commit()

    def link_workflow(self, bot_id: int, workflow_id: int) -> None:
        self.conn.execute(
            "INSERT INTO bot_workflow(bot_id, workflow_id) VALUES (?, ?)",
            (bot_id, workflow_id),
        )
        self.conn.commit()

    def link_enhancement(self, bot_id: int, enhancement_id: int) -> None:
        self.conn.execute(
            "INSERT INTO bot_enhancement(bot_id, enhancement_id) VALUES (?, ?)",
            (bot_id, enhancement_id),
        )
        self.conn.commit()

    def _insert_menace(
        self,
        rec: BotRecord,
        models: Iterable[int],
        workflows: Iterable[int],
        enhancements: Iterable[int],
        *,
        source_menace_id: str | None = None,
    ) -> None:
        if not self.menace_db:
            return
        mdb = self.menace_db
        bot_id = rec.bid
        menace_id = source_menace_id or (
            router.menace_id if router else os.getenv("MENACE_ID", "")
        )
        with mdb.engine.begin() as conn:
            conn.execute(
                mdb.bots.insert().values(
                    bot_id=bot_id,
                    bot_name=rec.name,
                    bot_type=rec.type_,
                    assigned_task=_serialize_list(rec.tasks),
                    parent_bot_id=int(rec.parent_id) if str(rec.parent_id).isdigit() else None,
                    dependencies=_serialize_list(rec.dependencies),
                    resource_estimates=_safe_json_dumps(rec.resources),
                    creation_date=rec.creation_date,
                    last_modification_date=rec.last_modification_date,
                    status=rec.status,
                    version=rec.version,
                    estimated_profit=rec.estimated_profit,
                    source_menace_id=menace_id,
                )
            )
            for mid in models:
                row = conn.execute(
                    mdb.models.select().where(mdb.models.c.model_id == mid)
                ).fetchone()
                if row:
                    conn.execute(mdb.bot_models.insert().values(bot_id=bot_id, model_id=mid))
                else:
                    warnings.warn(f"invalid model_id {mid}")
            for wid in workflows:
                row = conn.execute(
                    mdb.workflows.select().where(mdb.workflows.c.workflow_id == wid)
                ).fetchone()
                if row:
                    conn.execute(mdb.bot_workflows.insert().values(bot_id=bot_id, workflow_id=wid))
                else:
                    warnings.warn(f"invalid workflow_id {wid}")
            for enh in enhancements:
                row = conn.execute(
                    mdb.enhancements.select().where(mdb.enhancements.c.enhancement_id == enh)
                ).fetchone()
                if row:
                    conn.execute(
                        mdb.bot_enhancements.insert().values(bot_id=bot_id, enhancement_id=enh)
                    )
                else:
                    warnings.warn(f"invalid enhancement_id {enh}")

    def migrate_to_menace(self) -> None:
        """Insert all stored bots into the configured MenaceDB."""
        if not self.menace_db:
            return
        for row in self.fetch_all():
            rec = BotRecord(
                name=row["name"],
                type_=row["type"],
                tasks=_deserialize_list(row["tasks"]),
                parent_id=row["parent_id"],
                dependencies=_deserialize_list(row["dependencies"]),
                resources=json.loads(row["resources"] or "{}"),
                hierarchy_level=row["hierarchy_level"],
                creation_date=row["creation_date"],
                last_modification_date=row["last_modification_date"],
                status=row["status"],
                version=row.get("version", ""),
                estimated_profit=row.get("estimated_profit", 0.0),
                bid=row["id"],
            )
            mids = [
                m[0]
                for m in self.conn.execute(
                    "SELECT model_id FROM bot_model WHERE bot_id=?",
                    (row["id"],),
                ).fetchall()
            ]
            wids = [
                w[0]
                for w in self.conn.execute(
                    "SELECT workflow_id FROM bot_workflow WHERE bot_id=?",
                    (row["id"],),
                ).fetchall()
            ]
            eids = [
                e[0]
                for e in self.conn.execute(
                    "SELECT enhancement_id FROM bot_enhancement WHERE bot_id=?",
                    (row["id"],),
                ).fetchall()
            ]
            try:
                self._insert_menace(
                    rec,
                    mids,
                    wids,
                    eids,
                    source_menace_id=row.get("source_menace_id")
                    or (router.menace_id if router else None),
                )
            except Exception as exc:
                logger.exception("MenaceDB insert failed for %s: %s", rec.name, exc)
                self.failed_menace.append(
                    FailedMenace(
                        rec,
                        mids,
                        wids,
                        eids,
                        router.menace_id if router else os.getenv("MENACE_ID", ""),
                    )
                )

    def close(self) -> None:
        self.conn.close()


__all__ = ["BotRecord", "BotDB"]
