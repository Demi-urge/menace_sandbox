from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from time import time

from .auto_link import auto_link

from .unified_event_bus import UnifiedEventBus
from .retry_utils import publish_with_retry
import warnings
try:
    from .menace_db import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore
    warnings.warn("MenaceDB unavailable, Menace integration disabled.")
from uuid import uuid4


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
    attempt: int = 1
    next_retry: float = field(default_factory=time)


class BotDB:
    """SQLite database tracking bots and relationships."""

    MAX_RETRIES = 5

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        menace_db: "MenaceDB" | None = None,
    ) -> None:
        db_path = path or os.getenv("BOT_DB_PATH", "bots.db")
        # make connection usable across threads for async tasks
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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
                creation_date TEXT,
                last_modification_date TEXT,
                status TEXT,
                version TEXT,
                estimated_profit REAL
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(bots)").fetchall()]
        if "hierarchy_level" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN hierarchy_level TEXT")
        if "version" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN version TEXT")
        if "estimated_profit" not in cols:
            self.conn.execute("ALTER TABLE bots ADD COLUMN estimated_profit REAL")
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

    # basic helpers -----------------------------------------------------
    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM bots WHERE LOWER(name)=LOWER(?)", (name,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def fetch_all(self) -> list[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM bots")
        return [dict(r) for r in cur.fetchall()]

    def by_level(self, level: str) -> list[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM bots WHERE hierarchy_level=?",
            (level,),
        )
        return [dict(r) for r in cur.fetchall()]

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
                    self._insert_menace(item.rec, item.models, item.workflows, item.enhancements)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.exception("MenaceDB insert retry failed for %s: %s", item.rec.name, exc)
                    item.attempt += 1
                    if item.attempt <= self.MAX_RETRIES:
                        item.next_retry = now + 2 ** (item.attempt - 1)
                        still_pending.append(item)
                
            self.failed_menace = still_pending

    @auto_link({"models": "link_model", "workflows": "link_workflow", "enhancements": "link_enhancement"})
    def add_bot(
        self,
        rec: BotRecord,
        *,
        models: Iterable[int] | None = None,
        workflows: Iterable[int] | None = None,
        enhancements: Iterable[int] | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO bots(
                name, type, tasks, parent_id, dependencies,
                resources, hierarchy_level,
                creation_date, last_modification_date, status,
                version, estimated_profit
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.name,
                rec.type_,
                _serialize_list(rec.tasks),
                rec.parent_id,
                _serialize_list(rec.dependencies),
                _safe_json_dumps(rec.resources),
                rec.hierarchy_level,
                rec.creation_date,
                rec.last_modification_date,
                rec.status,
                rec.version,
                rec.estimated_profit,
            ),
        )
        rec.bid = int(cur.lastrowid)
        self.conn.commit()
        if self.menace_db:
            try:
                self._insert_menace(rec, models or [], workflows or [], enhancements or [])
            except Exception as exc:
                logger.exception("MenaceDB insert failed: %s", exc)
                self.failed_menace.append(
                    FailedMenace(rec, models or [], workflows or [], enhancements or [])
                )
        if self.event_bus:
            payload = dataclasses.asdict(rec)
            if not publish_with_retry(self.event_bus, "bot:new", payload):
                logger.exception("failed to publish bot:new event")
                self.failed_events.append(FailedEvent("bot:new", payload))
        return rec.bid

    def update_bot(self, bot_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["last_modification_date"] = datetime.utcnow().isoformat()
        sets = ", ".join(f"{k}=?" for k in fields)
        params = list(fields.values()) + [bot_id]
        self.conn.execute(f"UPDATE bots SET {sets} WHERE id=?", params)
        self.conn.commit()
        if self.event_bus:
            payload = {"bot_id": bot_id, **fields}
            if not publish_with_retry(self.event_bus, "bot:update", payload):
                logger.exception("failed to publish bot:update event")
                self.failed_events.append(FailedEvent("bot:update", payload))

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
    ) -> None:
        if not self.menace_db:
            return
        mdb = self.menace_db
        bot_id = rec.bid
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
            mids = [m[0] for m in self.conn.execute("SELECT model_id FROM bot_model WHERE bot_id=?", (row["id"],)).fetchall()]
            wids = [w[0] for w in self.conn.execute("SELECT workflow_id FROM bot_workflow WHERE bot_id=?", (row["id"],)).fetchall()]
            eids = [e[0] for e in self.conn.execute("SELECT enhancement_id FROM bot_enhancement WHERE bot_id=?", (row["id"],)).fetchall()]
            try:
                self._insert_menace(rec, mids, wids, eids)
            except Exception as exc:
                logger.exception("MenaceDB insert failed for %s: %s", rec.name, exc)
                self.failed_menace.append(FailedMenace(rec, mids, wids, eids))

    def close(self) -> None:
        self.conn.close()


__all__ = ["BotRecord", "BotDB"]
