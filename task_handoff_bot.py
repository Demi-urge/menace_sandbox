"""Task Handoff Bot for packaging and sending tasks to Stage 4."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Any, Iterable, Optional
import logging

import sqlite3
from .unified_event_bus import UnifiedEventBus
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


class WorkflowDB:
    """SQLite storage for generated workflows."""

    def __init__(self, path: Path = Path("workflows.db"), *, event_bus: Optional[UnifiedEventBus] = None) -> None:
        self.path = path
        self.event_bus = event_bus
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow TEXT,
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
                """
            )
            conn.commit()

    def update_status(self, workflow_id: int, status: str) -> None:
        """Update the status for a single workflow."""
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE workflows SET status=? WHERE id=?",
                (status, workflow_id),
            )
            conn.commit()

    def update_statuses(self, workflow_ids: Iterable[int], status: str) -> None:
        """Bulk update workflow status."""
        ids = list(workflow_ids)
        with sqlite3.connect(self.path) as conn:
            for wid in ids:
                conn.execute(
                    "UPDATE workflows SET status=? WHERE id=?",
                    (status, wid),
                )
            conn.commit()
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

    def add(self, wf: WorkflowRecord) -> int:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                """
                INSERT INTO workflows(
                    workflow, assigned_bots, enhancements, title, description,
                    task_sequence, tags, category, type, status,
                    rejection_reason, workflow_duration, performance_data, estimated_profit_per_bot, timestamp
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ",".join(wf.workflow),
                    ",".join(wf.assigned_bots),
                    ",".join(wf.enhancements),
                    wf.title,
                    wf.description,
                    ",".join(wf.task_sequence),
                    ",".join(wf.tags),
                    wf.category,
                    wf.type_,
                    wf.status,
                    wf.rejection_reason,
                    wf.workflow_duration,
                    wf.performance_data,
                    wf.estimated_profit_per_bot,
                    wf.timestamp,
                ),
            )
            conn.commit()
            wf.wid = cur.lastrowid
            if self.event_bus:
                try:
                    self.event_bus.publish("workflows:new", asdict(wf))
                except Exception as exc:
                    logger.warning(
                        "failed to publish new workflow %s: %s",
                        wf.wid,
                        exc,
                    )
            return wf.wid

    def fetch(self, limit: int = 20) -> List[WorkflowRecord]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM workflows ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results: List[WorkflowRecord] = []
        for row in rows:
            results.append(
                WorkflowRecord(
                    workflow=row["workflow"].split(",") if row["workflow"] else [],
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
            )
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
