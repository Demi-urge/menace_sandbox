"""Task Handoff Bot for packaging and sending tasks to Stage 4."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Any, Iterable, Optional, Sequence, Iterator
import logging

import sqlite3
from .unified_event_bus import UnifiedEventBus
from .workflow_graph import WorkflowGraph
from vector_service import EmbeddableDBMixin
from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from db_dedup import insert_if_unique
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zmq = None  # type: ignore
try:
    import pika  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pika = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Detailed information about a task."""

    name: str
    dependencies: List[str]
    resources: Dict[str, float]
    schedule: str
    code: str
    metadata: Dict[str, Any]


@dataclass
class TaskPackage:
    """Package of tasks ready for Stage 4."""

    tasks: List[TaskInfo]
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class WorkflowRecord:
    """Stored workflow broken down from a plan."""

    workflow: List[str]
    action_chains: List[str] = field(default_factory=list)
    argument_strings: List[str] = field(default_factory=list)
    assigned_bots: List[str] = field(default_factory=list)
    enhancements: List[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    task_sequence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: str = ""
    type_: str = ""
    status: str = "pending"
    rejection_reason: str = ""
    workflow_duration: float = 0.0
    performance_data: str = ""
    estimated_profit_per_bot: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    wid: int = 0



class WorkflowDB(EmbeddableDBMixin):
    """SQLite storage for generated workflows with vector search."""

    def __init__(
        self,
        path: Path | str = Path("workflows.db"),
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        workflow_graph: Optional[WorkflowGraph] = None,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "workflow_embeddings.index",
        embedding_version: int = 1,
        router: DBRouter | None = None,
    ) -> None:
        self.path = path
        self.event_bus = event_bus
        self.graph = workflow_graph
        self.vector_backend = vector_backend  # kept for compatibility
        LOCAL_TABLES.add("workflows")
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "task_handoff", local_db_path=str(self.path), shared_db_path=str(self.path)
        )
        self.conn = self.router.get_connection("workflows")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow TEXT,
                action_chains TEXT,
                argument_strings TEXT,
                assigned_bots TEXT,
                enhancements TEXT,
                title TEXT,
                description TEXT,
                task_sequence TEXT,
                tags TEXT,
                category TEXT,
                type TEXT,
                status TEXT,
                rejection_reason TEXT,
                workflow_duration REAL,
                performance_data TEXT,
                estimated_profit_per_bot REAL,
                timestamp TEXT
            )
            """,
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(workflows)").fetchall()]
        if "action_chains" not in cols:
            self.conn.execute("ALTER TABLE workflows ADD COLUMN action_chains TEXT")
        if "argument_strings" not in cols:
            self.conn.execute("ALTER TABLE workflows ADD COLUMN argument_strings TEXT")
        if "content_hash" not in cols:
            try:
                self.conn.execute(
                    "ALTER TABLE workflows ADD COLUMN content_hash TEXT UNIQUE"
                )
            except sqlite3.OperationalError:
                self.conn.execute(
                    "ALTER TABLE workflows ADD COLUMN content_hash TEXT"
                )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_workflows_content_hash ON workflows(content_hash)"
        )
        self.conn.commit()
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    # --------------------------------------------------------------
    # helpers
    def _row_to_record(self, row: sqlite3.Row) -> WorkflowRecord:
        return WorkflowRecord(
            workflow=row["workflow"].split(",") if row["workflow"] else [],
            action_chains=row["action_chains"].split(",") if row["action_chains"] else [],
            argument_strings=row["argument_strings"].split(",") if row["argument_strings"] else [],
            assigned_bots=row["assigned_bots"].split(",") if row["assigned_bots"] else [],
            enhancements=row["enhancements"].split(",") if row["enhancements"] else [],
            title=row["title"] or "",
            description=row["description"] or "",
            task_sequence=row["task_sequence"].split(",") if row["task_sequence"] else [],
            tags=row["tags"].split(",") if row["tags"] else [],
            category=row["category"] or "",
            type_=row["type"] or "",
            status=row["status"] or "",
            rejection_reason=row["rejection_reason"] or "",
            workflow_duration=row["workflow_duration"] or 0.0,
            performance_data=row["performance_data"] or "",
            estimated_profit_per_bot=row["estimated_profit_per_bot"] or 0.0,
            timestamp=row["timestamp"],
            wid=row["id"],
        )

    def _vector_text(self, rec: WorkflowRecord) -> str:
        """Build a descriptive text representation of ``rec``."""

        actions = rec.workflow or rec.task_sequence
        sequence = rec.task_sequence if rec.workflow != rec.task_sequence else []
        parts: list[str] = []
        if actions:
            parts.append(" -> ".join(actions))
        if sequence:
            parts.append("sequence: " + " -> ".join(sequence))
        if rec.argument_strings:
            parts.append("args: " + ", ".join(rec.argument_strings))
        return " | ".join(parts)

    def usage_rate(self, workflow_id: int) -> int:
        """Return count of bots using a workflow.

        Prefers the ``assigned_bots`` list stored on the workflow record. If
        absent, falls back to counting relationships in ``bot_workflow`` when
        that table is present in the connected database.
        """
        row = self.conn.execute(
            "SELECT assigned_bots FROM workflows WHERE id=?",
            (workflow_id,),
        ).fetchone()
        if row and row["assigned_bots"]:
            bots = [b for b in row["assigned_bots"].split(",") if b]
            return len(bots)
        try:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM bot_workflow WHERE workflow_id=?",
                (workflow_id,),
            )
            count = cur.fetchone()
            return int(count[0]) if count else 0
        except sqlite3.Error:
            return 0

    # --------------------------------------------------------------
    # status updates
    def update_status(self, workflow_id: int, status: str) -> None:
        """Update the status for a single workflow and refresh embedding."""
        self.conn.execute(
            "UPDATE workflows SET status=? WHERE id=?",
            (status, workflow_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM workflows WHERE id=?",
            (workflow_id,),
        ).fetchone()
        if row:
            rec = self._row_to_record(row)
            try:
                self.add_embedding(workflow_id, rec, "workflow", source_id=str(workflow_id))
            except Exception as exc:  # pragma: no cover - best effort
                logger.exception(
                    "embedding hook failed for %s: %s", workflow_id, exc
                )
        if self.event_bus:
            try:
                payload = {"workflow_id": workflow_id, "status": status}
                self.event_bus.publish("workflows:update", payload)
            except Exception as exc:
                logger.warning(
                    "failed to publish workflow status %s for %s: %s",
                    status,
                    workflow_id,
                    exc,
                )
        if self.graph:
            try:
                self.graph.update(str(workflow_id), "update")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("failed to update workflow graph for %s: %s", workflow_id, exc)

    def update_statuses(self, workflow_ids: Iterable[int], status: str) -> None:
        """Bulk update workflow status and refresh embeddings."""
        ids = list(workflow_ids)
        for wid in ids:
            self.conn.execute(
                "UPDATE workflows SET status=? WHERE id=?",
                (status, wid),
            )
        self.conn.commit()
        for wid in ids:
            row = self.conn.execute(
                "SELECT * FROM workflows WHERE id=?",
                (wid,),
            ).fetchone()
            if row:
                rec = self._row_to_record(row)
                try:
                    self.add_embedding(wid, rec, "workflow", source_id=str(wid))
                except Exception as exc:  # pragma: no cover - best effort
                    logger.exception("embedding hook failed for %s: %s", wid, exc)
                if self.graph:
                    try:
                        self.graph.update(str(wid), "update")
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.warning("failed to update workflow graph for %s: %s", wid, exc)
        if self.event_bus:
            try:
                for wid in ids:
                    payload = {"workflow_id": wid, "status": status}
                    self.event_bus.publish("workflows:update", payload)
            except Exception as exc:
                logger.warning(
                    "failed to publish workflow status %s for %s: %s",
                    status,
                    ids,
                    exc,
                )

    # --------------------------------------------------------------
    # insert/fetch
    def add(self, wf: WorkflowRecord, source_menace_id: str = "") -> int:
        values = {
            "workflow": ",".join(wf.workflow),
            "action_chains": ",".join(wf.action_chains),
            "argument_strings": ",".join(wf.argument_strings),
            "assigned_bots": ",".join(wf.assigned_bots),
            "enhancements": ",".join(wf.enhancements),
            "title": wf.title,
            "description": wf.description,
            "task_sequence": ",".join(wf.task_sequence),
            "tags": ",".join(wf.tags),
            "category": wf.category,
            "type": wf.type_,
            "status": wf.status,
            "rejection_reason": wf.rejection_reason,
            "workflow_duration": wf.workflow_duration,
            "performance_data": wf.performance_data,
            "estimated_profit_per_bot": wf.estimated_profit_per_bot,
            "timestamp": wf.timestamp,
        }
        hash_fields = [
            "workflow",
            "action_chains",
            "argument_strings",
            "title",
            "description",
            "task_sequence",
        ]
        inserted = insert_if_unique(
            "workflows",
            values,
            hash_fields,
            source_menace_id,
            self.router,
        )
        cur = self.conn.execute(
            "SELECT id FROM workflows WHERE content_hash=?",
            (values["content_hash"],),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("failed to retrieve workflow id")
        wf.wid = int(row["id"])
        if not inserted:
            logger.warning(
                "duplicate workflow detected; skipping embedding generation"
            )
            return wf.wid

        try:
            self.add_embedding(wf.wid, wf, "workflow", source_id=str(wf.wid))
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s: %s", wf.wid, exc)

        if self.event_bus:
            try:
                self.event_bus.publish("workflows:new", asdict(wf))
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception as exc:
                logger.warning(
                    "failed to publish new workflow %s: %s",
                    wf.wid,
                    exc,
                )
        if self.graph:
            try:
                self.graph.update(str(wf.wid), "add")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("failed to update workflow graph for %s: %s", wf.wid, exc)
        return wf.wid

    def remove(self, workflow_id: int) -> None:
        """Delete a workflow and emit an event."""
        self.conn.execute("DELETE FROM workflows WHERE id=?", (workflow_id,))
        self.conn.commit()
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "workflows:deleted", {"workflow_id": workflow_id}
                )
            except Exception as exc:
                logger.warning(
                    "failed to publish workflow deletion %s: %s", workflow_id, exc
                )
        if self.graph:
            try:
                self.graph.update(str(workflow_id), "remove")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "failed to update workflow graph for deletion %s: %s", workflow_id, exc
                )

    def replace(self, workflow_id: int, wf: WorkflowRecord) -> int:
        """Replace an existing workflow with ``wf``.

        The original workflow is removed and a new one inserted; the identifier
        of the new workflow is returned.
        """

        self.remove(workflow_id)
        return self.add(wf)

    def fetch(self, limit: int = 20) -> List[WorkflowRecord]:
        self.conn.row_factory = sqlite3.Row
        rows = self.conn.execute(
            "SELECT * FROM workflows ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: List[WorkflowRecord] = []
        for row in rows:
            results.append(self._row_to_record(row))
        return results

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(self) -> Iterator[tuple[int, WorkflowRecord, str]]:
        """Yield all workflow records for embedding backfill."""
        cur = self.conn.execute("SELECT * FROM workflows")
        for row in cur.fetchall():
            rec = self._row_to_record(row)
            yield rec.wid, rec, "workflow"

    # --------------------------------------------------------------
    # embedding/search
    def vector(self, rec: Any) -> list[float]:
        """Embed ``rec`` into a vector using its action chain and arguments."""

        if isinstance(rec, (int, str)):
            row = self.conn.execute(
                "SELECT * FROM workflows WHERE id=?",
                (int(rec),),
            ).fetchone()
            if not row:
                raise ValueError("record not found")
            rec = self._row_to_record(row)
        elif isinstance(rec, sqlite3.Row):
            rec = self._row_to_record(rec)
        elif not isinstance(rec, WorkflowRecord):
            raise TypeError("unsupported record type")
        text = self._vector_text(rec)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Encode ``text`` to a vector (overridable for tests)."""
        return self.encode_text(text)

    def search_by_vector(self, vector: Iterable[float], top_k: int = 5) -> List[WorkflowRecord]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[WorkflowRecord] = []
        for rec_id, dist in matches:
            row = self.conn.execute(
                "SELECT * FROM workflows WHERE id=?",
                (rec_id,),
            ).fetchone()
            if row:
                rec = self._row_to_record(row)
                setattr(rec, "_distance", dist)
                results.append(rec)
        return results


class TaskHandoffBot:
    """Compile tasks, record workflows and transmit them to Stage 4."""

    def __init__(
        self,
        api_url: str = "http://localhost:9000/handoff",
        mq_url: str | None = None,
        pair_addr: str | None = None,
        workflow_db: Optional[WorkflowDB] = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
    ) -> None:
        self.api_url = api_url
        if zmq:
            self.context = zmq.Context.instance()
            self.socket = self.context.socket(zmq.PAIR)
            self.addr = pair_addr or f"inproc://handoff-{uuid.uuid4().hex}"
            self.socket.bind(self.addr)
            self._noblock = zmq.NOBLOCK
            self._again = zmq.Again
            self._error = zmq.ZMQError
        else:  # pragma: no cover - zmq unavailable
            from multiprocessing.connection import Listener, Client, Connection
            import os

            class _IpcSocket:
                """Simple IPC socket using :mod:`multiprocessing.connection`."""

                def __init__(self) -> None:
                    self._listener: Listener | None = None
                    self._conn: Connection | None = None
                    self._address: str | None = None

                def bind(self, address: str) -> None:  # noqa: D401 - bind address
                    self._address = address or f"/tmp/handoff-{uuid.uuid4().hex}.sock"
                    if os.path.exists(self._address):
                        try:
                            os.unlink(self._address)
                        except OSError:
                            pass
                    self._listener = Listener(self._address, family="AF_UNIX")

                def connect(self, address: str) -> None:
                    self._address = address
                    self._conn = Client(address, family="AF_UNIX")

                def _ensure_conn(self) -> None:
                    if self._conn is None and self._listener is not None:
                        try:
                            self._conn = self._listener.accept()
                        except Exception as exc:  # pragma: no cover - accept error
                            raise _Again("no connection") from exc

                def send_json(self, msg: dict, *a, flags: int | None = None, **k) -> None:
                    self._ensure_conn()
                    if self._conn is None:
                        raise _Again("not connected")
                    try:
                        self._conn.send(msg)
                    except Exception as exc:
                        if flags is not None:
                            raise _Again() from exc
                        raise

                def recv_json(self, *a, flags: int | None = None, **k) -> dict:
                    self._ensure_conn()
                    if self._conn is None:
                        raise _Again("not connected")
                    if flags is not None and not self._conn.poll():
                        raise _Again()
                    try:
                        return self._conn.recv()
                    except Exception as exc:
                        if flags is not None:
                            raise _Again() from exc
                        raise

                def close(self) -> None:
                    if self._conn is not None:
                        try:
                            self._conn.close()
                        except Exception:
                            pass
                        self._conn = None
                    if self._listener is not None:
                        try:
                            self._listener.close()
                        except Exception:
                            pass
                        if isinstance(self._listener.address, str):
                            try:
                                os.unlink(self._listener.address)
                            except Exception:
                                pass
                        self._listener = None

            class _Again(Exception):
                """Raised when a non-blocking IPC operation would block."""
                pass

            self.context = None
            self.socket = _IpcSocket()
            self.addr = pair_addr or f"/tmp/handoff-{uuid.uuid4().hex}.sock"
            self.socket.bind(self.addr)
            self._noblock = 0
            self._again = _Again
            self._error = Exception
        self.channel = None
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        if mq_url and pika:
            try:
                params = pika.URLParameters(mq_url)
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.queue_declare(queue="handoff", durable=True)
                self.channel = ch
            except Exception:
                self.channel = None

    def compile(self, tasks: List[TaskInfo]) -> TaskPackage:
        return TaskPackage(tasks=tasks)

    def _split_tasks(self, names: List[str], min_size: int = 3) -> List[List[str]]:
        chunks = [names]
        size = len(names)
        while size > min_size:
            size = max(min_size, size // 2)
            for i in range(0, len(names), size):
                chunk = names[i : i + size]
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
        return chunks

    def store_plan(
        self,
        tasks: Iterable[TaskInfo],
        enhancements: Iterable[str] | None = None,
        title: str = "",
        description: str = "",
    ) -> List[int]:
        names = [t.name for t in tasks]
        ids: List[int] = []
        for chunk in self._split_tasks(names):
            rec = WorkflowRecord(
                workflow=chunk,
                title=title,
                description=description,
                task_sequence=chunk,
                enhancements=list(enhancements or []),
            )
            ids.append(self.workflow_db.add(rec))
        return ids

    # --------------------------------------------------------------
    # Status helpers
    # --------------------------------------------------------------

    def mark_workflows(self, workflow_ids: Iterable[int], status: str) -> None:
        """Update status for a collection of workflows."""
        ids = list(workflow_ids)
        try:
            self.workflow_db.update_statuses(ids, status)
            logger.info("marked workflows %s as %s", ids, status)
        except Exception as exc:
            logger.warning(
                "failed to update status %s for %s: %s", status, ids, exc
            )

    def mark_active(self, workflow_ids: Iterable[int]) -> None:
        self.mark_workflows(workflow_ids, "active")

    def mark_failed(self, workflow_ids: Iterable[int]) -> None:
        self.mark_workflows(workflow_ids, "failed")

    def send_package(self, package: TaskPackage) -> None:
        data = package.to_json()
        try:
            requests.post(self.api_url, data=data, headers={"Content-Type": "application/json"}, timeout=3)
        except Exception as exc:
            logger.warning("primary handoff failed: %s", exc)
            if self.channel:
                try:
                    self.channel.basic_publish("", "handoff", data)
                except Exception as mq_exc:
                    logger.warning("mq publish failed: %s", mq_exc)

    def respond_to_queries(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        try:
            msg = self.socket.recv_json(flags=self._noblock)
        except self._again:
            return
        resp = handler(msg)
        try:
            self.socket.send_json(resp, flags=self._noblock)
        except self._error as exc:
            logger.warning("failed to send response: %s", exc)

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception as exc:
            logger.warning("failed to close handoff socket: %s", exc)


__all__ = [
    "TaskInfo",
    "TaskPackage",
    "WorkflowRecord",
    "WorkflowDB",
    "TaskHandoffBot",
]
