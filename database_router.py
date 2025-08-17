from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
import json
import base64
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable
import sqlite3
import logging

try:  # pragma: no cover - optional dependency
    from . import cross_query
except Exception:  # pragma: no cover - missing optional dependency
    cross_query = None  # type: ignore

from . import database_manager as dm

from .code_database import CodeDB, CodeRecord
from .bot_database import BotDB, BotRecord
from .transaction_manager import TransactionManager
from .access_control import READ, WRITE, ADMIN, check_permission
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - circular imports
    from .bot_registry import BotRegistry
    from .neuroplasticity import PathwayDB
    from .research_aggregator_bot import InfoDB, ResearchItem
    from .error_bot import ErrorDB
    from .chatgpt_enhancement_bot import EnhancementDB
else:  # pragma: no cover - type checking only
    InfoDB = object  # type: ignore
    ResearchItem = object  # type: ignore
    BotRegistry = object  # type: ignore
    PathwayDB = object  # type: ignore
    ErrorDB = object  # type: ignore
    EnhancementDB = object  # type: ignore
from .menace_memory_manager import MenaceMemoryManager
from .unified_event_bus import UnifiedEventBus
from .task_handoff_bot import WorkflowDB, WorkflowRecord
from .audit_trail import AuditTrail
from vector_service import EmbeddableDBMixin, Retriever, FallbackResult
try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when missing
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector service lacks explicit class."""

        pass
try:
    from .databases import MenaceDB
except Exception:  # pragma: no cover - optional dependency
    MenaceDB = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class DBResult:
    code: List[Dict[str, Any]]
    bots: List[Dict[str, Any]]
    info: List[Any]
    memory: List[Any]
    menace: List[Dict[str, Any]]


class _QueryEncoder(EmbeddableDBMixin):
    """Lightweight encoder used for semantic search queries."""

    def __init__(self) -> None:  # pragma: no cover - simple initialiser
        tmp = Path(tempfile.gettempdir())
        super().__init__(index_path=tmp / "router_query.index", metadata_path=tmp / "router_query.json")

    def vector(self, record: Any) -> List[float]:  # pragma: no cover - unused
        raise NotImplementedError

    def iter_records(self):  # pragma: no cover - unused
        return iter(())


class DatabaseRouter:
    """Aggregate read and write access across Menace databases.

    The router exposes convenience helpers for reading from the local SQLite
    stores and for writing new objects.  When a writable :class:`MenaceDB`
    instance is supplied the write helpers will mirror inserts to that
    database as well.
    """

    def __init__(
        self,
        code_db: Optional[CodeDB] = None,
        bot_db: Optional[BotDB] = None,
        info_db: Optional[InfoDB] = None,
        memory_mgr: Optional[MenaceMemoryManager] = None,
        workflow_db: Optional[WorkflowDB] = None,
        menace_db: Optional[MenaceDB] = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        auto_cross_link: bool = True,
        remote_url: str | None = None,
        transaction_manager: "TransactionManager" | None = None,
        cache_seconds: float = 60.0,
        bot_roles: Optional[Dict[str, str]] = None,
        audit_trail_path: str | None = None,
        audit_privkey: bytes | None = None,
        error_db: "ErrorDB" | None = None,
        enhancement_db: "EnhancementDB" | None = None,
        min_reliability: float | None = None,
        redundancy_limit: int | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        from .research_aggregator_bot import InfoDB as _InfoDB
        from .error_bot import ErrorDB as _ErrorDB
        from .chatgpt_enhancement_bot import EnhancementDB as _EnhancementDB

        self.code_db = code_db or CodeDB(event_bus=event_bus)
        self.bot_db = bot_db or BotDB(event_bus=event_bus)
        self.info_db = info_db or _InfoDB(event_bus=event_bus)
        self.memory_mgr = memory_mgr or MenaceMemoryManager(event_bus=event_bus, bot_db=self.bot_db, info_db=self.info_db)
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        self.error_db = error_db or _ErrorDB(event_bus=event_bus)
        self.enhancement_db = enhancement_db or _EnhancementDB()
        self.remote_url = remote_url
        if menace_db is not None:
            self.menace_db = menace_db
        elif remote_url and MenaceDB is not None:
            try:
                self.menace_db = MenaceDB(url=remote_url)
            except Exception as exc:
                logger.error("Failed to connect to remote MenaceDB: %s", exc)
                self.menace_db = None
        elif MenaceDB is not None:
            try:
                self.menace_db = MenaceDB()
            except Exception as exc:
                logger.error("Failed to open local MenaceDB: %s", exc)
                self.menace_db = None
        else:
            self.menace_db = None

        path = audit_trail_path or os.getenv("AUDIT_LOG_PATH", "audit.log")
        key_b64 = audit_privkey or os.getenv("AUDIT_PRIVKEY")
        # If no key is available create an unsigned trail with a warning
        if key_b64:
            priv = base64.b64decode(key_b64) if isinstance(key_b64, str) else key_b64
        else:
            logger.warning(
                "AUDIT_PRIVKEY not set; audit trail entries will not be signed"
            )
            priv = None
        self.audit_trail = AuditTrail(path, priv)

        self.transaction_manager = transaction_manager or TransactionManager()

        self._local_checksums: list[tuple[str, str]] = []

        self.bot_roles: Dict[str, str] = bot_roles or {}

        self.event_bus = event_bus
        if auto_cross_link and event_bus:
            event_bus.subscribe("info:new", self._on_info_new)
            event_bus.subscribe("bot:new", self._on_bot_new)
            event_bus.subscribe("code:new", self._on_code_new)
            event_bus.subscribe("memory:new", self._on_memory_new)

        self.cache_seconds = cache_seconds
        self.cache_enabled = cache_seconds > 0
        self._cache: Dict[tuple, tuple[float, Any]] = {}
        self._query_encoder = _QueryEncoder()
        self._retriever = retriever

        env_min_rel = float(os.getenv("DB_MIN_RELIABILITY", "0.0"))
        env_redundancy = int(os.getenv("DB_REDUNDANCY_LIMIT", "1"))
        self.min_reliability = (
            float(min_reliability)
            if min_reliability is not None
            else env_min_rel
        )
        self.redundancy_limit = (
            int(redundancy_limit)
            if redundancy_limit is not None
            else env_redundancy
        )

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def _check_permission(self, action: str, requesting_bot: str | None) -> None:
        if not requesting_bot:
            return
        role = self.bot_roles.get(requesting_bot, READ)
        check_permission(role, action)

    def _log_action(self, requesting_bot: str | None, action: str, details: dict) -> None:
        bot = requesting_bot or "unknown"
        ts = datetime.utcnow().isoformat()
        try:
            payload = json.dumps({"timestamp": ts, "bot": bot, "action": action, "details": details}, sort_keys=True)
            self.audit_trail.record(payload)
        except Exception as exc:
            logger.error("audit trail write failed: %s", exc)
        if not self.menace_db:
            return
        try:
            with self.menace_db.engine.begin() as conn:
                conn.execute(
                    self.menace_db.audit_log.insert().values(
                        timestamp=ts,
                        bot_name=bot,
                        action=action,
                        details=json.dumps(details),
                    )
                )
        except Exception as exc:
            logger.error("audit DB write failed: %s", exc)

    def _get_cache(self, key: tuple) -> Any:
        if not self.cache_enabled:
            return None
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self.cache_seconds:
            del self._cache[key]
            return None
        return value

    def _set_cache(self, key: tuple, value: Any) -> None:
        if self.cache_enabled:
            self._cache[key] = (time.time(), value)

    def _record_checksum(self, data: dict) -> None:
        """Store a checksum for replicated transaction."""
        if not self.remote_url or not self.menace_db:
            return
        try:
            payload = json.dumps(data, sort_keys=True)
        except Exception as exc:
            logger.error("checksum serialization failed: %s", exc)
            payload = str(data)
        checksum = hashlib.sha256(payload.encode()).hexdigest()
        ts = datetime.utcnow().isoformat()
        self._local_checksums.append((ts, checksum))
        try:
            with self.menace_db.engine.begin() as conn:
                conn.execute(
                    self.menace_db.replication_checksums.insert().values(
                        timestamp=ts,
                        checksum=checksum,
                    )
                )
        except Exception as exc:
            logger.error("checksum write failed: %s", exc)

    def _run_with_checksum(
        self, operation: Callable[[], Any], rollback: Callable[[], None], data: dict
    ) -> Any:
        res = self.transaction_manager.run(operation, rollback)
        self._record_checksum(data)
        return res

    # ------------------------------------------------------------------
    # Cache management helpers
    # ------------------------------------------------------------------

    def flush_cache(self) -> None:
        """Clear all cached query results."""
        self._cache.clear()

    def enable_cache(self, enabled: bool) -> None:
        """Enable or disable in-memory caching."""
        self.cache_enabled = enabled
        if not enabled:
            self.flush_cache()

    def _redundant_retrieve(self, query: Any, top_k: int) -> List[Dict[str, Any]]:
        if self._retriever is None:
            return []
        try:
            hits = self._retriever.search(query, top_k=top_k)
            if isinstance(hits, (FallbackResult, ErrorResult)):
                if isinstance(hits, FallbackResult):
                    logger.debug(
                        "retriever returned fallback for %s: %s",
                        query,
                        getattr(hits, "reason", ""),
                    )
                return []
        except Exception as exc:
            logger.error("retrieval failed: %s", exc)
            return []
        return hits[:top_k]

    def query_all(self, term: str, *, requesting_bot: str | None = None) -> DBResult:
        self._check_permission(READ, requesting_bot)
        term_l = term.lower()
        try:
            code = [c for c in self.code_db.fetch_all() if term_l in (c.get("summary", "") + c.get("code", "")).lower()]
        except Exception as exc:
            logger.error("code search failed: %s", exc)
            code = []
        try:
            bots = [b for b in self.bot_db.fetch_all() if term_l in b.get("name", "").lower()]
        except Exception as exc:
            logger.error("bot search failed: %s", exc)
            bots = []
        try:
            info = self.info_db.search(term)
        except Exception as exc:
            logger.error("info search failed: %s", exc)
            info = []
        try:
            memory = self.memory_mgr.search_by_tag(term)
        except Exception as exc:
            logger.error("memory search failed: %s", exc)
            memory = []
        if self.menace_db:
            try:
                menace = self.menace_db.search(term)
            except Exception as exc:
                logger.error("menace search failed: %s", exc)
                menace = []
        else:
            menace = []
        result = DBResult(code=code, bots=bots, info=info, memory=memory, menace=menace)
        self._log_action(requesting_bot, "query_all", {"term": term})
        return result

    def universal_search(self, query: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search across configured databases using the shared retriever."""

        try:
            return self._redundant_retrieve(query, top_k)
        except Exception as exc:
            logger.error("universal retrieval failed: %s", exc)
            return []

    def semantic_search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform a semantic search via :class:`Retriever`."""

        try:
            hits = self._redundant_retrieve(query_text, top_k)
        except Exception as exc:
            logger.error("semantic retrieval failed: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for h in hits:
            item = dict(h)
            origin = item.get("origin_db")
            if origin == "information":
                item["origin_db"] = "info"
            results.append(item)

        return results[:top_k]

    def execute_query(
        self, db: str, query: str, params: Iterable[Any] | None = None, *, requesting_bot: str | None = None
    ) -> List[Dict[str, Any]]:
        """Execute *query* against one of the router's SQLite databases."""
        self._check_permission(READ, requesting_bot)
        params = params or []
        key = ("execute_query", db, query, tuple(params))
        cached = self._get_cache(key)
        if cached is not None:
            return cached
        conn = None
        close_after = False
        if db == "code":
            path = getattr(self.code_db, "path", None)
            if path:
                conn = sqlite3.connect(path)
                close_after = True
        elif db == "bot":
            conn = getattr(self.bot_db, "conn", None)
        elif db == "info":
            path = getattr(self.info_db, "path", None)
            if path:
                conn = sqlite3.connect(path)
                close_after = True
        elif db == "memory":
            conn = getattr(self.memory_mgr, "conn", None)
        else:
            raise ValueError(f"Unknown database {db}")
        if conn is None:
            raise ValueError(f"No connection for {db}")
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        if close_after:
            conn.close()
        self._set_cache(key, rows)
        self._log_action(requesting_bot, "execute_query", {"db": db, "query": query})
        return rows

    def _publish_change(self, table: str, action: str, payload: Dict[str, Any]) -> None:
        if self.event_bus:
            try:
                self.event_bus.publish(f"cdc:{table}", {"action": action, **payload})
            except Exception as exc:
                logger.error("event publish failed: %s", exc)

    def existing_code(self, name: str) -> Optional[str]:
        res = self.query_all(name)
        for entry in res.code:
            if name.lower() in (entry.get("summary", "") + entry.get("code", "")).lower():
                return entry.get("code", "")
        return None

    # ------------------------------------------------------------------
    # Cross query wrappers
    # ------------------------------------------------------------------

    def related_workflows(
        self,
        bot_name: str,
        *,
        registry: BotRegistry,
        pathway_db: PathwayDB | None = None,
        depth: int = 1,
    ) -> List[str]:
        """Wrapper around :func:`cross_query.related_workflows`."""
        if not self.menace_db or cross_query is None:
            return []
        key = (
            "related_workflows",
            bot_name,
            id(registry),
            id(pathway_db),
            depth,
        )
        cached = self._get_cache(key)
        if cached is not None:
            return cached
        res = cross_query.related_workflows(
            bot_name,
            registry=registry,
            menace_db=self.menace_db,
            pathway_db=pathway_db,
            depth=depth,
        )
        self._set_cache(key, res)
        return res

    def similar_code_snippets(
        self,
        template: str,
        *,
        registry: BotRegistry | None = None,
        pathway_db: PathwayDB | None = None,
        limit: int = 5,
    ) -> List[Dict[str, object]]:
        """Wrapper around :func:`cross_query.similar_code_snippets`."""
        if not self.menace_db or cross_query is None:
            return []
        key = (
            "similar_code_snippets",
            template,
            id(registry),
            id(pathway_db),
            limit,
        )
        cached = self._get_cache(key)
        if cached is not None:
            return cached
        res = cross_query.similar_code_snippets(
            template,
            menace_db=self.menace_db,
            registry=registry,
            pathway_db=pathway_db,
            limit=limit,
        )
        self._set_cache(key, res)
        return res

    def related_resources(
        self,
        bot_name: str,
        *,
        registry: BotRegistry,
        pathway_db: PathwayDB | None = None,
        depth: int = 1,
    ) -> Dict[str, List[str]]:
        """Wrapper around :func:`cross_query.related_resources`."""
        if not self.menace_db or cross_query is None:
            return {
                "bots": [],
                "workflows": [],
                "information": [],
                "memory": [],
            }
        key = (
            "related_resources",
            bot_name,
            id(registry),
            id(pathway_db),
            depth,
        )
        cached = self._get_cache(key)
        if cached is not None:
            return cached
        res = cross_query.related_resources(
            bot_name,
            registry=registry,
            menace_db=self.menace_db,
            info_db=self.info_db,
            memory_mgr=self.memory_mgr,
            pathway_db=pathway_db,
            depth=depth,
        )
        self._set_cache(key, res)
        return res

    # ------------------------------------------------------------------
    # Write-through helpers
    # ------------------------------------------------------------------

    def insert_bot(
        self,
        rec: "BotRecord",
        *,
        model_ids: Iterable[int] | None = None,
        workflow_ids: Iterable[int] | None = None,
        enhancement_ids: Iterable[int] | None = None,
        requesting_bot: str | None = None,
    ) -> int:
        """Store a new bot record in local and Menace databases."""

        self._check_permission(WRITE, requesting_bot)

        bid = self.bot_db.add_bot(rec)
        if self.menace_db:
            bot_id_int = bid
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.bots.insert().values(
                            bot_id=bot_id_int,
                            bot_name=rec.name,
                            bot_type=rec.type_,
                            assigned_task=",".join(rec.tasks),
                            parent_bot_id=int(rec.parent_id, 16)
                            if rec.parent_id
                            else None,
                            dependencies=",".join(rec.dependencies),
                            resource_estimates=json.dumps(rec.resources),
                            creation_date=rec.creation_date,
                            last_modification_date=rec.last_modification_date,
                            status=rec.status,
                            version="",
                            estimated_profit=0.0,
                        )
                    )
                    for mid in model_ids or []:
                        conn.execute(
                            self.menace_db.bot_models.insert().values(
                                bot_id=bot_id_int, model_id=mid
                            )
                        )
                    for wid in workflow_ids or []:
                        conn.execute(
                            self.menace_db.workflow_bots.insert().values(
                                workflow_id=wid, bot_id=bot_id_int
                            )
                        )
                    for enh in enhancement_ids or []:
                        conn.execute(
                            self.menace_db.bot_enhancements.insert().values(
                                bot_id=bot_id_int, enhancement_id=enh
                            )
                        )

            def rollback() -> None:
                try:
                    self.bot_db.delete_bot(bid)
                except Exception as exc:
                    logger.error("rollback delete_bot failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "insert_bot", "bot_id": bid},
                )
                self._publish_change("bots", "insert", {"bot_id": bid})
            except Exception as exc:
                logger.error("insert_bot failed: %s", exc)
                raise
        self._log_action(requesting_bot, "insert_bot", {"bot_id": bid})
        return bid

    def insert_model(self, name: str, *, requesting_bot: str | None = None, **fields: Any) -> int:
        """Insert a new model and mirror to MenaceDB."""
        self._check_permission(WRITE, requesting_bot)
        mid = dm.add_model(name, **fields)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.models.insert().values(
                            model_id=mid,
                            model_name=name,
                            source=fields.get("source", ""),
                            date_discovered=datetime.utcnow().isoformat(),
                            tags=fields.get("tags", ""),
                            initial_roi_prediction=fields.get(
                                "initial_roi_prediction", 0.0
                            ),
                            final_roi_prediction=fields.get(
                                "final_roi_prediction", 0.0
                            ),
                            current_status=fields.get("current_status", ""),
                            enhancement_count=0,
                            discrepancy_flag=False,
                            error_flag=False,
                            profitability_score=fields.get(
                                "profitability_score", 0.0
                            ),
                        )
                    )
                    if fields.get("workflow_id") is not None:
                        conn.execute(
                            self.menace_db.model_workflows.insert().values(
                                model_id=mid, workflow_id=fields["workflow_id"]
                            )
                        )

            def rollback() -> None:
                try:
                    self.delete_model(mid)
                except Exception as exc:
                    logger.error("rollback delete_model failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "insert_model", "model_id": mid},
                )
                self._publish_change("models", "insert", {"model_id": mid})
            except Exception as exc:
                logger.error("insert_model failed: %s", exc)
                raise
        self._log_action(requesting_bot, "insert_model", {"model_id": mid})
        return mid

    def insert_workflow(self, wf: "WorkflowRecord", *, requesting_bot: str | None = None) -> int:
        """Insert a workflow and mirror to MenaceDB."""

        self._check_permission(WRITE, requesting_bot)
        wid = self.workflow_db.add(wf) if hasattr(self, "workflow_db") else 0
        if self.menace_db and wid:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.workflows.insert().values(
                            workflow_id=wid,
                            workflow_name=wf.title,
                            task_tree=",".join(wf.workflow),
                            dependencies="",
                            resource_allocation_plan="",
                            created_from="",
                            enhancement_links=",".join(wf.enhancements),
                            discrepancy_links="",
                            status=wf.status,
                            estimated_profit_per_bot=wf.estimated_profit_per_bot,
                        )
                    )
                    for bid in wf.assigned_bots:
                        try:
                            conn.execute(
                                self.menace_db.workflow_bots.insert().values(
                                    workflow_id=wid, bot_id=bid
                                )
                            )
                        except Exception as exc:
                            logger.error("workflow bot link failed: %s", exc)

            def rollback() -> None:
                try:
                    if hasattr(self.workflow_db, "delete"):
                        self.workflow_db.delete(wid)
                    else:
                        import sqlite3

                        path = getattr(self.workflow_db, "path", None)
                        if path:
                            with sqlite3.connect(path) as conn:
                                conn.execute("DELETE FROM workflows WHERE id=?", (wid,))
                                conn.commit()
                except Exception as exc:
                    logger.error("workflow rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "insert_workflow", "workflow_id": wid},
                )
                self._publish_change("workflows", "insert", {"workflow_id": wid})
            except Exception as exc:
                logger.error("insert_workflow failed: %s", exc)
                raise
        self._log_action(requesting_bot, "insert_workflow", {"workflow_id": wid})
        return wid

    def insert_info(
        self,
        item: "ResearchItem",
        *,
        workflows: Iterable[int] | None = None,
        enhancements: Iterable[int] | None = None,
        requesting_bot: str | None = None,
    ) -> int:
        """Insert a research item and mirror to MenaceDB."""
        self._check_permission(WRITE, requesting_bot)
        info_id = self.info_db.add(
            item, workflows=workflows, enhancements=enhancements
        )
        if self.menace_db and not getattr(self.info_db, "menace_db", None):
            old = getattr(self.info_db, "menace_db", None)

            def op() -> None:
                self.info_db.menace_db = self.menace_db
                self.info_db._insert_menace(item, workflows or [])

            def rollback() -> None:
                try:
                    if hasattr(self.info_db, "delete"):
                        self.info_db.delete(info_id)
                    else:
                        import sqlite3

                        path = getattr(self.info_db, "path", None)
                        if path:
                            with sqlite3.connect(path) as conn:
                                conn.execute("DELETE FROM info WHERE id=?", (info_id,))
                                conn.commit()
                except Exception as exc:
                    logger.error("info rollback failed: %s", exc)
                finally:
                    self.info_db.menace_db = old

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "insert_info", "info_id": info_id},
                )
                self._publish_change(
                    "information", "insert", {"info_id": info_id}
                )
            except Exception as exc:
                logger.error("insert_info failed: %s", exc)
                raise
            finally:
                self.info_db.menace_db = old
        self._log_action(requesting_bot, "insert_info", {"info_id": info_id})
        return info_id

    def insert_code(self, rec: "CodeRecord", *, requesting_bot: str | None = None) -> int:
        """Insert a code template and mirror to MenaceDB."""
        self._check_permission(WRITE, requesting_bot)
        cid = self.code_db.add(rec)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.code.insert().values(
                            code_id=cid,
                            template_type=rec.template_type,
                            language=rec.language,
                            version=rec.version,
                            complexity_score=rec.complexity_score,
                            code_summary=rec.summary,
                        )
                    )

            def rollback() -> None:
                try:
                    if hasattr(self.code_db, "delete"):
                        self.code_db.delete(cid)
                    else:
                        import sqlite3

                        path = getattr(self.code_db, "path", None)
                        if path:
                            with sqlite3.connect(path) as conn:
                                conn.execute("DELETE FROM code WHERE id=?", (cid,))
                                conn.commit()
                except Exception as exc:
                    logger.error("code rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "insert_code", "code_id": cid},
                )
                self._publish_change("code", "insert", {"code_id": cid})
            except Exception as exc:
                logger.error("insert_code failed: %s", exc)
                raise
        self._log_action(requesting_bot, "insert_code", {"code_id": cid})
        return cid

    # ------------------------------------------------------------------
    # Update/delete helpers
    # ------------------------------------------------------------------

    def update_bot(self, bot_id: int, *, requesting_bot: str | None = None, **fields: Any) -> None:
        """Update a bot and mirror changes."""
        self._check_permission(WRITE, requesting_bot)
        if not fields:
            return
        cur = self.bot_db.conn.execute("SELECT * FROM bots WHERE id=?", (bot_id,))
        row = cur.fetchone()
        self.bot_db.update_bot(bot_id, **fields)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.bots.update()
                        .where(self.menace_db.bots.c.bot_id == bot_id)
                        .values(**fields)
                    )

            def rollback() -> None:
                if row:
                    try:
                        self.bot_db.conn.execute(
                            "UPDATE bots SET name=?, type=?, tasks=?, parent_id=?, dependencies=?, resources=?, hierarchy_level=?, creation_date=?, last_modification_date=?, status=? WHERE id=?",
                            (
                                row["name"],
                                row["type"],
                                row["tasks"],
                                row["parent_id"],
                                row["dependencies"],
                                row["resources"],
                                row["hierarchy_level"],
                                row["creation_date"],
                                row["last_modification_date"],
                                row["status"],
                                bot_id,
                            ),
                        )
                        self.bot_db.conn.commit()
                    except Exception as exc:
                        logger.error("update_bot rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "update_bot", "bot_id": bot_id, **fields},
                )
                self._publish_change("bots", "update", {"bot_id": bot_id, **fields})
            except Exception as exc:
                logger.error("update_bot failed: %s", exc)
                raise
        self._log_action(requesting_bot, "update_bot", {"bot_id": bot_id})

    def delete_bot(self, bot_id: int, *, requesting_bot: str | None = None) -> None:
        self._check_permission(WRITE, requesting_bot)
        cur = self.bot_db.conn.execute("SELECT * FROM bots WHERE id=?", (bot_id,))
        row = cur.fetchone()
        try:
            self.bot_db.conn.execute("DELETE FROM bots WHERE id=?", (bot_id,))
            self.bot_db.conn.commit()
        except Exception as exc:
            logger.error("local bot delete failed: %s", exc)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.bots.delete().where(
                            self.menace_db.bots.c.bot_id == bot_id
                        )
                    )

            def rollback() -> None:
                if row:
                    try:
                        self.bot_db.conn.execute(
                            "INSERT INTO bots(id,name,type,tasks,parent_id,dependencies,resources,hierarchy_level,creation_date,last_modification_date,status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                row["id"],
                                row["name"],
                                row["type"],
                                row["tasks"],
                                row["parent_id"],
                                row["dependencies"],
                                row["resources"],
                                row["hierarchy_level"],
                                row["creation_date"],
                                row["last_modification_date"],
                                row["status"],
                            ),
                        )
                        self.bot_db.conn.commit()
                    except Exception as exc:
                        logger.error("delete_bot rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "delete_bot", "bot_id": bot_id},
                )
                self._publish_change("bots", "delete", {"bot_id": bot_id})
            except Exception as exc:
                logger.error("delete_bot failed: %s", exc)
                raise
        self._log_action(requesting_bot, "delete_bot", {"bot_id": bot_id})

    def update_model(self, model_id: int, *, requesting_bot: str | None = None, **fields: Any) -> None:
        self._check_permission(WRITE, requesting_bot)
        if not fields:
            return
        import sqlite3

        with sqlite3.connect(dm.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            old_row = conn.execute(
                "SELECT * FROM models WHERE id=?", (model_id,)
            ).fetchone()
        dm.update_model(model_id, **fields)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.models.update()
                        .where(self.menace_db.models.c.model_id == model_id)
                        .values(**fields)
                    )

            def rollback() -> None:
                if old_row:
                    try:
                        cols = [
                            "name",
                            "source",
                            "date_discovered",
                            "tags",
                            "roi_metadata",
                            "exploration_status",
                            "profitability_score",
                            "current_roi",
                            "final_roi_prediction",
                            "initial_roi_prediction",
                            "current_status",
                            "workflow_id",
                        ]
                        sets = ", ".join(f"{c}=?" for c in cols)
                        params = [old_row[c] for c in cols] + [model_id]
                        with sqlite3.connect(dm.DB_PATH) as conn:
                            conn.execute(
                                f"UPDATE models SET {sets} WHERE id=?",
                                params,
                            )
                            conn.commit()
                    except Exception as exc:
                        logger.error("update_model rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "update_model", "model_id": model_id, **fields},
                )
                self._publish_change("models", "update", {"model_id": model_id, **fields})
            except Exception as exc:
                logger.error("update_model failed: %s", exc)
                raise
        self._log_action(requesting_bot, "update_model", {"model_id": model_id})

    def delete_model(self, model_id: int, *, requesting_bot: str | None = None) -> None:
        self._check_permission(WRITE, requesting_bot)
        import sqlite3

        with sqlite3.connect(dm.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        dm.delete_model(model_id)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.models.delete().where(
                            self.menace_db.models.c.model_id == model_id
                        )
                    )

            def rollback() -> None:
                if row:
                    try:
                        cols = [
                            "id",
                            "name",
                            "source",
                            "date_discovered",
                            "tags",
                            "roi_metadata",
                            "exploration_status",
                            "profitability_score",
                            "current_roi",
                            "final_roi_prediction",
                            "initial_roi_prediction",
                            "current_status",
                            "workflow_id",
                        ]
                        placeholders = ",".join("?" for _ in cols)
                        with sqlite3.connect(dm.DB_PATH) as conn:
                            conn.execute(
                                f"INSERT INTO models({', '.join(cols)}) VALUES ({placeholders})",
                                [row[c] for c in cols],
                            )
                            conn.commit()
                    except Exception as exc:
                        logger.error("delete_model rollback insert failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "delete_model", "model_id": model_id},
                )
                self._publish_change("models", "delete", {"model_id": model_id})
            except Exception as exc:
                logger.error("delete_model failed: %s", exc)
                raise
        self._log_action(requesting_bot, "delete_model", {"model_id": model_id})

    def update_code(self, code_id: int, *, requesting_bot: str | None = None, **fields: Any) -> None:
        self._check_permission(WRITE, requesting_bot)
        if not fields:
            return
        import sqlite3

        path = getattr(self.code_db, "path", None)
        row = None
        if path:
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM code WHERE id=?", (code_id,)).fetchone()
        self.code_db.update(code_id, **fields)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.code.update()
                        .where(self.menace_db.code.c.code_id == code_id)
                        .values(**fields)
                    )

            def rollback() -> None:
                if row is not None and path:
                    try:
                        sets = ", ".join(f"{k}=?" for k in row.keys() if k != "id")
                        params = [row[k] for k in row.keys() if k != "id"] + [code_id]
                        with sqlite3.connect(path) as conn:
                            conn.execute(f"UPDATE code SET {sets} WHERE id=?", params)
                            conn.commit()
                    except Exception as exc:
                        logger.error("update_code rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "update_code", "code_id": code_id, **fields},
                )
                self._publish_change("code", "update", {"code_id": code_id, **fields})
            except Exception as exc:
                logger.error("update_code failed: %s", exc)
                raise
        self._log_action(requesting_bot, "update_code", {"code_id": code_id})

    def update_workflow(self, workflow_id: int, *, requesting_bot: str | None = None, **fields: Any) -> None:
        self._check_permission(WRITE, requesting_bot)
        if hasattr(self.workflow_db, "update") and fields:
            import sqlite3

            path = getattr(self.workflow_db, "path", None)
            row = None
            if path:
                try:
                    with sqlite3.connect(path) as conn:
                        conn.row_factory = sqlite3.Row
                        row = conn.execute(
                            "SELECT * FROM workflows WHERE id=?", (workflow_id,)
                        ).fetchone()
                except Exception as exc:
                    logger.error("workflow lookup failed: %s", exc)
                    row = None
            try:
                self.workflow_db.update(workflow_id, **fields)
            except Exception as exc:
                logger.error("workflow update local failed: %s", exc)
                row = None
            if self.menace_db:
                def op() -> None:
                    with self.menace_db.engine.begin() as conn:
                        conn.execute(
                            self.menace_db.workflows.update()
                            .where(self.menace_db.workflows.c.workflow_id == workflow_id)
                            .values(**fields)
                        )

                def rollback() -> None:
                    if row is not None and path:
                        try:
                            sets = ", ".join(f"{k}=?" for k in row.keys() if k != "id")
                            params = [row[k] for k in row.keys() if k != "id"] + [workflow_id]
                            with sqlite3.connect(path) as conn:
                                conn.execute(
                                    f"UPDATE workflows SET {sets} WHERE id=?",
                                    params,
                                )
                                conn.commit()
                        except Exception as exc:
                            logger.error("update_workflow rollback failed: %s", exc)

                try:
                    self._run_with_checksum(
                        op,
                        rollback,
                        {"action": "update_workflow", "workflow_id": workflow_id, **fields},
                    )
                    self._publish_change(
                        "workflows", "update", {"workflow_id": workflow_id, **fields}
                    )
                except Exception as exc:
                    logger.error("update_workflow failed: %s", exc)
                    raise
        self._log_action(requesting_bot, "update_workflow", {"workflow_id": workflow_id})

    def update_info(self, info_id: int, *, requesting_bot: str | None = None, **fields: Any) -> None:
        """Update an info item in local and Menace databases."""
        self._check_permission(WRITE, requesting_bot)
        if not fields:
            return
        import sqlite3

        path = getattr(self.info_db, "path", None)
        row = None
        if path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT * FROM info WHERE id=?", (info_id,)).fetchone()
            except Exception as exc:
                logger.error("info lookup failed: %s", exc)
                row = None
        if hasattr(self.info_db, "update"):
            try:
                self.info_db.update(info_id, **fields)
            except Exception as exc:
                logger.error("info update local failed: %s", exc)
                row = None
        elif path:
            try:
                sets = ", ".join(f"{k}=?" for k in fields)
                params = list(fields.values()) + [info_id]
                with sqlite3.connect(path) as conn:
                    conn.execute(f"UPDATE info SET {sets} WHERE id=?", params)
                    conn.commit()
            except Exception as exc:
                logger.error("info update sqlite failed: %s", exc)
                row = None
        if self.menace_db:
            emb = None
            try:
                emb = self.info_db.vector(info_id)  # type: ignore[attr-defined]
            except Exception:
                emb = None

            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.information.update()
                        .where(self.menace_db.information.c.info_id == info_id)
                        .values(**fields)
                    )
                    if emb is not None:
                        conn.execute(
                            self.menace_db.information_embeddings.insert().prefix_with("OR REPLACE").values(
                                record_id=str(info_id),
                                vector=json.dumps(emb),
                                created_at=datetime.utcnow().isoformat(),
                                embedding_version=getattr(self.info_db, "embedding_version", 1),
                                kind="info",
                                source_id=str(info_id),
                            )
                        )

            def rollback() -> None:
                if row is not None and path:
                    try:
                        sets = ", ".join(f"{k}=?" for k in row.keys() if k != "id")
                        params = [row[k] for k in row.keys() if k != "id"] + [info_id]
                        with sqlite3.connect(path) as conn:
                            conn.execute(f"UPDATE info SET {sets} WHERE id=?", params)
                            conn.commit()
                    except Exception as exc:
                        logger.error("update_info rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "update_info", "info_id": info_id, **fields},
                )
                self._publish_change(
                    "information", "update", {"info_id": info_id, **fields}
                )
            except Exception as exc:
                logger.error("update_info failed: %s", exc)
                raise
        self._log_action(requesting_bot, "update_info", {"info_id": info_id})

    def delete_info(self, info_id: int, *, requesting_bot: str | None = None) -> None:
        """Delete an info item from local and Menace databases."""
        self._check_permission(WRITE, requesting_bot)
        import sqlite3

        path = getattr(self.info_db, "path", None)
        row = None
        if path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT * FROM info WHERE id=?", (info_id,)).fetchone()
            except Exception as exc:
                logger.error("info lookup failed: %s", exc)
                row = None
        if hasattr(self.info_db, "delete"):
            try:
                self.info_db.delete(info_id)
            except Exception as exc:
                logger.error("info delete local failed: %s", exc)
        elif path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.execute("DELETE FROM info WHERE id=?", (info_id,))
                    conn.commit()
            except Exception as exc:
                logger.error("info delete sqlite failed: %s", exc)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.information.delete().where(
                            self.menace_db.information.c.info_id == info_id
                        )
                    )

            def rollback() -> None:
                if row is not None and path:
                    try:
                        cols = [k for k in row.keys()]
                        placeholders = ",".join("?" for _ in cols)
                        with sqlite3.connect(path) as conn:
                            conn.execute(
                                f"INSERT INTO info({', '.join(cols)}) VALUES ({placeholders})",
                                [row[c] for c in cols],
                            )
                            conn.commit()
                    except Exception as exc:
                        logger.error("delete_info rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "delete_info", "info_id": info_id},
                )
                self._publish_change("information", "delete", {"info_id": info_id})
            except Exception as exc:
                logger.error("delete_info failed: %s", exc)
                raise
        self._log_action(requesting_bot, "delete_info", {"info_id": info_id})

    def delete_workflow(self, workflow_id: int, *, requesting_bot: str | None = None) -> None:
        """Remove a workflow from local storage and MenaceDB."""
        self._check_permission(WRITE, requesting_bot)
        import sqlite3

        path = getattr(self.workflow_db, "path", None)
        row = None
        if path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
            except Exception as exc:
                logger.error("workflow lookup failed: %s", exc)
                row = None
        if hasattr(self.workflow_db, "delete"):
            try:
                self.workflow_db.delete(workflow_id)
            except Exception as exc:
                logger.error("workflow local delete failed: %s", exc)
        elif path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.execute("DELETE FROM workflows WHERE id=?", (workflow_id,))
                    conn.commit()
            except Exception as exc:
                logger.error("workflow sqlite delete failed: %s", exc)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.workflows.delete().where(
                            self.menace_db.workflows.c.workflow_id == workflow_id
                        )
                    )

            def rollback() -> None:
                if row is not None and path:
                    try:
                        cols = [k for k in row.keys()]
                        placeholders = ",".join("?" for _ in cols)
                        with sqlite3.connect(path) as conn:
                            conn.execute(
                                f"INSERT INTO workflows({', '.join(cols)}) VALUES ({placeholders})",
                                [row[c] for c in cols],
                            )
                            conn.commit()
                    except Exception as exc:
                        logger.error("delete_workflow rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "delete_workflow", "workflow_id": workflow_id},
                )
                self._publish_change(
                    "workflows", "delete", {"workflow_id": workflow_id}
                )
            except Exception as exc:
                logger.error("delete_workflow failed: %s", exc)
                raise
        self._log_action(requesting_bot, "delete_workflow", {"workflow_id": workflow_id})

    def delete_code(self, code_id: int, *, requesting_bot: str | None = None) -> None:
        """Delete a code template from local storage and MenaceDB."""

        self._check_permission(WRITE, requesting_bot)
        import sqlite3

        path = getattr(self.code_db, "path", None)
        row = None
        if path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT * FROM code WHERE id=?", (code_id,)).fetchone()
            except Exception as exc:
                logger.error("code lookup failed: %s", exc)
                row = None
        if hasattr(self.code_db, "delete"):
            try:
                self.code_db.delete(code_id)
            except Exception as exc:
                logger.error("code delete local failed: %s", exc)
        elif path:
            try:
                with sqlite3.connect(path) as conn:
                    conn.execute("DELETE FROM code WHERE id=?", (code_id,))
                    conn.commit()
            except Exception as exc:
                logger.error("code delete sqlite failed: %s", exc)
        if self.menace_db:
            def op() -> None:
                with self.menace_db.engine.begin() as conn:
                    conn.execute(
                        self.menace_db.code.delete().where(
                            self.menace_db.code.c.code_id == code_id
                        )
                    )

            def rollback() -> None:
                if row is not None and path:
                    try:
                        cols = [k for k in row.keys()]
                        placeholders = ",".join("?" for _ in cols)
                        with sqlite3.connect(path) as conn:
                            conn.execute(
                                f"INSERT INTO code({', '.join(cols)}) VALUES ({placeholders})",
                                [row[c] for c in cols],
                            )
                            conn.commit()
                    except Exception as exc:
                        logger.error("delete_code rollback failed: %s", exc)

            try:
                self._run_with_checksum(
                    op,
                    rollback,
                    {"action": "delete_code", "code_id": code_id},
                )
                self._publish_change("code", "delete", {"code_id": code_id})
            except Exception as exc:
                logger.error("delete_code failed: %s", exc)
                raise
        self._log_action(requesting_bot, "delete_code", {"code_id": code_id})

    def verify_replication(self) -> bool:
        """Return True if local and remote replication checksums match."""
        if not self.remote_url or not self.menace_db:
            return True
        try:
            with self.menace_db.engine.begin() as conn:
                rows = conn.execute(
                    self.menace_db.replication_checksums.select()
                ).mappings().fetchall()
            remote = [(r["timestamp"], r["checksum"]) for r in rows]
            return remote == self._local_checksums
        except Exception as exc:
            logger.error("replication check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_info_new(self, topic: str, payload: object) -> None:
        try:
            data = payload if isinstance(payload, dict) else payload.__dict__
            info_id = int(data.get("item_id"))
            bots = data.get("associated_bots", [])
            for bname in bots:
                rec = self.bot_db.find_by_name(bname)
                if rec:
                    try:
                        self.info_db.link_bot(info_id, rec["id"])
                    except Exception as exc:
                        logger.error("link bot failed: %s", exc)
        except Exception as exc:
            logger.error("info new handler failed: %s", exc)

    def _on_bot_new(self, topic: str, payload: object) -> None:
        try:
            data = payload if isinstance(payload, dict) else payload.__dict__
            name = data.get("name") or data.get("bot_name")
            bot_id = int(data.get("id") or data.get("bot_id"))
            if not name:
                return
            try:
                items = self.info_db.search(name)
            except Exception as exc:
                logger.error("info search failed: %s", exc)
                items = []
            for it in items:
                iid = getattr(it, "item_id", None)
                if iid is not None:
                    try:
                        self.info_db.link_bot(iid, bot_id)
                    except Exception as exc:
                        logger.error("link bot info failed: %s", exc)
            try:
                mems = self.memory_mgr.search_by_tag(name)
            except Exception as exc:
                logger.error("memory search failed: %s", exc)
                mems = []
            for m in mems:
                try:
                    self.memory_mgr.conn.execute(
                        "UPDATE memory SET bot_id=? WHERE key=? AND version=?",
                        (bot_id, m.key, m.version),
                    )
                except Exception as exc:
                    logger.error("memory update failed: %s", exc)
            self.memory_mgr.conn.commit()
        except Exception as exc:
            logger.error("bot new handler failed: %s", exc)

    def _on_code_new(self, topic: str, payload: object) -> None:
        try:
            data = payload if isinstance(payload, dict) else payload.__dict__
            cid = int(data.get("cid") or data.get("code_id"))
            text = " ".join(str(data.get(f, "")) for f in ["summary", "template_type"])
            bots = []
            for b in self.bot_db.fetch_all():
                if b.get("name") and b["name"] in text:
                    bots.append(b)
            for b in bots:
                try:
                    self.code_db.link_bot(cid, str(b["id"]))
                except Exception as exc:
                    logger.error("code link failed: %s", exc)
        except Exception as exc:
            logger.error("code new handler failed: %s", exc)

    def _on_memory_new(self, topic: str, payload: object) -> None:
        try:
            data = payload if isinstance(payload, dict) else payload.__dict__
            tags = data.get("tags", "")
            key = data.get("key")
            version = data.get("version")
            if not key or version is None:
                return
            for tag in str(tags).split():
                try:
                    rec = self.bot_db.find_by_name(tag)
                    if rec:
                        self.memory_mgr.conn.execute(
                            "UPDATE memory SET bot_id=? WHERE key=? AND version=?",
                            (rec["id"], key, version),
                        )
                except Exception as exc:
                    logger.error("memory bot link failed: %s", exc)
                try:
                    items = self.info_db.search(tag)
                except Exception as exc:
                    logger.error("memory info search failed: %s", exc)
                    items = []
                for it in items:
                    iid = getattr(it, "item_id", None)
                    if iid is not None:
                        try:
                            self.memory_mgr.conn.execute(
                                "UPDATE memory SET info_id=? WHERE key=? AND version=?",
                                (iid, key, version),
                            )
                        except Exception as exc:
                            logger.error("memory info link failed: %s", exc)
            self.memory_mgr.conn.commit()
        except Exception as exc:
            logger.error("memory new handler failed: %s", exc)

__all__ = ["DatabaseRouter", "DBResult", "verify_replication"]
