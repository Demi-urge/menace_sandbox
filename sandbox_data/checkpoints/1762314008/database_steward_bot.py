"""Database Steward Bot for Stage 5 data management."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Iterable

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
    from sqlalchemy.exc import SQLAlchemyError  # type: ignore
    from sqlalchemy.orm import sessionmaker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Column = Integer = MetaData = String = Table = create_engine = Engine = SQLAlchemyError = sessionmaker = None  # type: ignore

try:
    import pymongo  # type: ignore
except Exception:  # pragma: no cover - optional
    pymongo = None  # type: ignore

try:
    from elasticsearch import Elasticsearch  # type: ignore
except Exception:  # pragma: no cover - optional
    Elasticsearch = None  # type: ignore

try:
    import git  # type: ignore
except Exception:  # pragma: no cover - optional
    git = None  # type: ignore

from .conversation_manager_bot import ConversationManagerBot
from .db_router import DBRouter
from .admin_bot_base import AdminBotBase

if TYPE_CHECKING:
    from .error_bot import ErrorBot


@dataclass
class Lock:
    """Record level lock info."""

    key: str
    token: str
    ts: float


class SQLStore:
    """Manage SQL templates table."""

    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        self.engine: Engine = create_engine(url)
        self.meta = MetaData()
        self.templates = Table(
            "templates",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("name", String),
            Column("version", Integer),
            Column("priority", Integer, default=0),
            Column("updated", String),
        )
        self.meta.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add(self, name: str, version: int = 1, priority: int = 0) -> int:
        now = datetime.utcnow().isoformat()
        with self.engine.begin() as conn:
            res = conn.execute(
                self.templates.insert().values(
                    name=name, version=version, priority=priority, updated=now
                )
            )
            return int(res.inserted_primary_key[0])

    def fetch(self) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(self.templates.select()).fetchall()
            return [dict(row._mapping) for row in rows]


class MongoStore:
    """Wrapper around pymongo collection."""

    def __init__(self, url: str = "mongodb://localhost:27017", db: str = "steward") -> None:
        if pymongo:
            self.client = pymongo.MongoClient(url)
            self.col = self.client[db]["records"]
        else:  # pragma: no cover - fallback
            self.client = None
            self.col = []  # type: ignore

    def insert(self, doc: Dict[str, Any]) -> None:
        if pymongo:
            self.col.insert_one(doc)
        else:  # pragma: no cover - fallback
            self.col.append(doc)

    def find(self) -> List[Dict[str, Any]]:
        if pymongo:
            return list(self.col.find({}, {"_id": 0}))
        return list(self.col)


class ESIndex:
    """Minimal Elasticsearch wrapper."""

    def __init__(self, url: str = "http://localhost:9200", index: str = "steward") -> None:
        if Elasticsearch:
            self.es = Elasticsearch(url)
            self.index = index
            try:
                resp = self.es.indices.create(index=index, ignore=400)  # type: ignore
                if isinstance(resp, dict) and resp.get("error"):
                    err_type = resp["error"].get("type")
                    if err_type != "resource_already_exists_exception":
                        logger.error("failed to create index: %s", resp)
            except Exception as exc:  # pragma: no cover - optional
                logger.exception("failed to create index: %s", exc)
        else:  # pragma: no cover - fallback
            self.es = None
            self.index = index
            self.docs: List[Dict[str, Any]] = []

    def add(self, doc_id: str, body: Dict[str, Any]) -> None:
        if Elasticsearch:
            self.es.index(index=self.index, id=doc_id, body=body)  # type: ignore
        else:  # pragma: no cover - fallback
            self.docs.append({"id": doc_id, **body})

    def search_all(self) -> List[Dict[str, Any]]:
        if Elasticsearch:
            res = self.es.search(index=self.index, body={"query": {"match_all": {}}})  # type: ignore
            hits = res["hits"]["hits"]
            return [h["_source"] for h in hits]
        return [d for d in self.docs]


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class DatabaseStewardBot(AdminBotBase):
    """Manage schemas and data lifecycles across databases."""

    def __init__(
        self,
        sql_url: str = "sqlite:///:memory:",
        repo_path: Path | None = None,
        error_bot: ErrorBot | None = None,
        conversation_bot: "ConversationManagerBot" | None = None,
        db_router: DBRouter | None = None,
    ) -> None:
        super().__init__(db_router=db_router)
        self.sql = SQLStore(sql_url)
        self.mongo = MongoStore()
        self.es = ESIndex()
        self.repo = git.Repo.init(repo_path or Path("repo")) if git else None
        self.error_bot = error_bot
        self.conversation_bot = conversation_bot
        self.safe_mode = False
        self.locks: Dict[str, Lock] = {}
        self.metrics: Dict[str, int] = {}
        self.threshold = 0.5
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DatabaseStewardBot")

    # ------------------------------------------------------------------
    # Fix ideation helpers
    # ------------------------------------------------------------------
    def ideate_schema_fix(self) -> List[str]:
        """Return expected schema columns to restore consistency."""
        expected = {"id", "name", "version", "priority", "updated"}
        cols = {c.name for c in self.sql.templates.columns}
        if cols == expected:
            return list(expected)
        return sorted(expected)

    def _check_safe(self) -> None:
        if self.error_bot and self.error_bot.db.is_safe_mode(self.__class__.__name__):
            self.safe_mode = True
            if self.conversation_bot:
                try:
                    self.conversation_bot.notify(
                        f"{self.__class__.__name__} blocked in safe mode"
                    )
                except Exception as exc:
                    logger.exception("notify failed: %s", exc)
            raise RuntimeError("safe mode active")

    # --- Versioning helpers ---
    def version_file(self, path: Path) -> str:
        """Commit file and return commit hash."""
        self._check_safe()
        if not self.repo:  # pragma: no cover - optional
            return ""
        self.repo.index.add([str(path)])
        commit = self.repo.index.commit(f"version {path.name}")
        return commit.hexsha

    # --- Locking ---
    def lock(self, key: str, token: str) -> bool:
        self._check_safe()
        now = datetime.utcnow().timestamp()
        if key in self.locks and now - self.locks[key].ts < 300:
            return False
        self.locks[key] = Lock(key, token, now)
        return True

    def unlock(self, key: str, token: str) -> None:
        self._check_safe()
        if key in self.locks and self.locks[key].token == token:
            del self.locks[key]

    # --- Auditing ---
    def audit(self) -> List[str]:
        self._check_safe()
        issues: List[str] = []
        expected = {"id", "name", "version", "priority", "updated"}
        cols = {c.name for c in self.sql.templates.columns}
        if cols != expected:
            issues.append("schema_drift")
        return issues

    # --- Monitoring ---
    def record_usage(self, template_id: int, error: bool = False) -> None:
        self._check_safe()
        key = f"t{template_id}"
        self.metrics[key] = self.metrics.get(key, 0) + 1
        if error:
            self.metrics["errors"] = self.metrics.get("errors", 0) + 1

    # --- Maintenance ---
    def deduplicate(self) -> None:
        self._check_safe()
        self.query("templates")
        seen: Dict[str, int] = {}
        rows = self.sql.fetch()
        for row in rows:
            h = hashlib.sha1(row["name"].encode()).hexdigest()
            if h in seen:
                # delete dup
                with self.sql.engine.begin() as conn:
                    conn.execute(self.sql.templates.delete().where(self.sql.templates.c.id == row["id"]))
            else:
                seen[h] = row["id"]

    def purge_unused(self, max_age: int = 30) -> None:
        self._check_safe()
        cutoff = datetime.utcnow().timestamp() - max_age
        with self.sql.engine.begin() as conn:
            conn.execute(
                self.sql.templates.delete().where(
                    self.sql.templates.c.updated < datetime.utcfromtimestamp(cutoff).isoformat()
                )
            )

    def report(self) -> Dict[str, Any]:
        self._check_safe()
        return {"usage": self.metrics}

    def feedback(self, score: float) -> None:
        self._check_safe()
        self.threshold = (self.threshold + score) / 2.0

    # ------------------------------------------------------------------
    # Error handling integration
    # ------------------------------------------------------------------

    def resolve_management_issues(self, admin_bot: Path) -> List[str]:
        self._check_safe()
        """Audit databases and request fixes for any issues found."""
        issues = self.audit()
        if not issues:
            return issues

        if "schema_drift" in issues:
            expected = set(self.ideate_schema_fix())
            if self.error_bot:
                self.error_bot.fix_admin_bot_schema(admin_bot, expected)
                try:
                    self.error_bot.db.log_discrepancy("schema_drift")
                except Exception as exc:
                    logger.exception("log discrepancy failed: %s", exc)
            else:
                self.logger.warning("Schema drift detected but no error bot set")
        return issues


__all__ = ["SQLStore", "MongoStore", "ESIndex", "DatabaseStewardBot", "Lock"]