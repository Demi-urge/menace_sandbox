"""Centralized, tamper-resistant audit logging for Security AI."""

from __future__ import annotations

import csv
import json
import logging
import queue
import random
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dynamic_path_router import resolve_dir
from db_router import GLOBAL_ROUTER, init_db_router

# Directory and default log paths
LOG_DIR = resolve_dir("logs")
JSONL_PATH = LOG_DIR / "audit_log.jsonl"
SQLITE_PATH = LOG_DIR / "audit_log.db"


logger = logging.getLogger(__name__)


def _ensure_log_dir() -> None:
    """Create the log directory if missing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def generate_event_id(event_type: str) -> str:
    """Return a unique event identifier."""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"{event_type}-{ts}-{suffix}"


def log_event(event_type: str, data: Dict[str, Any], jsonl_path: Path = JSONL_PATH) -> str:
    """Append an event record to the JSONL audit log."""
    _ensure_log_dir()
    event_id = generate_event_id(event_type)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "event_id": event_id,
        "data": data,
    }
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return event_id


def export_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Convert a JSONL audit log to a flat CSV file."""
    if not jsonl_path.exists():
        return
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            flat: Dict[str, Any] = {}
            for k, v in entry.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v)
                else:
                    flat[k] = v
            rows.append(flat)
    if not rows:
        return
    headers = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _ensure_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not already exist."""

    with closing(conn.cursor()) as cur:
        cur.execute(
            """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS event_data (
                event_id TEXT,
                key TEXT,
                value TEXT
            )"""
        )
    conn.commit()


_SQLITE_RETRY_MESSAGES = (
    "database is locked",
    "database schema is locked",
    "cannot commit transaction - sql statements in progress",
)
_SQLITE_MAX_RETRIES = 5
_SQLITE_BASE_DELAY = 0.05
_SQLITE_TIMEOUT_SECONDS = 30.0


@dataclass(slots=True)
class _AuditPayload:
    """Container for an audit record awaiting persistence."""

    event_id: str
    event_type: str
    timestamp: str
    data: Dict[str, Any]


def _should_retry_sqlite(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return any(fragment in message for fragment in _SQLITE_RETRY_MESSAGES)


def _connection_database_path(conn: sqlite3.Connection) -> str:
    """Return the on-disk path for *conn*'s ``main`` database."""

    with closing(conn.execute("PRAGMA database_list")) as cursor:
        for _, name, path in cursor.fetchall():
            if name == "main":
                return path or ":memory:"
    return ":memory:"


def _coerce_value(value: Any) -> str:
    """Convert *value* into a SQLite-friendly string."""

    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _write_event(db_path: str, payload: _AuditPayload) -> None:
    """Persist *payload* into *db_path* within an isolated connection."""

    with sqlite3.connect(
        db_path,
        timeout=_SQLITE_TIMEOUT_SECONDS,
        isolation_level=None,
    ) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA journal_mode=DELETE")
        _ensure_db(conn)
        with conn:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    "INSERT INTO events (event_id, timestamp, event_type) VALUES (?, ?, ?)",
                    (payload.event_id, payload.timestamp, payload.event_type),
                )
                for key, value in payload.data.items():
                    cur.execute(
                        "INSERT INTO event_data (event_id, key, value) VALUES (?, ?, ?)",
                        (payload.event_id, key, _coerce_value(value)),
                    )


def _write_event_with_retry(db_path: str, payload: _AuditPayload) -> None:
    """Persist *payload* retrying transient SQLite failures using fresh connections."""

    last_error: Exception | None = None
    for attempt in range(_SQLITE_MAX_RETRIES):
        try:
            _write_event(db_path, payload)
            return
        except sqlite3.OperationalError as exc:
            last_error = exc
            if attempt < _SQLITE_MAX_RETRIES - 1 and _should_retry_sqlite(exc):
                delay = _SQLITE_BASE_DELAY * (2**attempt)
                time.sleep(delay)
                continue
            raise
    if last_error is not None:
        raise last_error


def _write_event_in_connection(conn: sqlite3.Connection, payload: _AuditPayload) -> None:
    """Persist *payload* using an existing in-memory connection."""

    _ensure_db(conn)
    try:
        with closing(conn.cursor()) as cur:
            cur.execute(
                "INSERT INTO events (event_id, timestamp, event_type) VALUES (?, ?, ?)",
                (payload.event_id, payload.timestamp, payload.event_type),
            )
            for key, value in payload.data.items():
                cur.execute(
                    "INSERT INTO event_data (event_id, key, value) VALUES (?, ?, ?)",
                    (payload.event_id, key, _coerce_value(value)),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class _SQLiteWorker:
    """Background worker that serialises writes to a SQLite database."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._queue: queue.Queue[tuple[_AuditPayload, queue.Queue[Exception | None]]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name=f"audit-sqlite-writer-{Path(db_path).stem or 'memory'}",
            daemon=True,
        )
        self._thread.start()

    def write(self, payload: _AuditPayload) -> None:
        """Synchronously persist *payload* using the worker thread."""

        result: queue.Queue[Exception | None] = queue.Queue(maxsize=1)
        self._queue.put((payload, result))
        error = result.get()
        if error is not None:
            raise error

    def _run(self) -> None:
        while True:
            payload, result = self._queue.get()
            try:
                _write_event_with_retry(self._db_path, payload)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to persist audit event %s: %s",
                    payload.event_id,
                    exc,
                )
                result.put(exc)
            else:
                result.put(None)


class _SQLiteWriterRegistry:
    """Lazy registry mapping database paths to writer threads."""

    def __init__(self) -> None:
        self._workers: Dict[str, _SQLiteWorker] = {}
        self._lock = threading.Lock()

    def write(self, db_path: str, payload: _AuditPayload) -> None:
        worker = self._get_worker(db_path)
        worker.write(payload)

    def _get_worker(self, db_path: str) -> _SQLiteWorker:
        with self._lock:
            worker = self._workers.get(db_path)
            if worker is None:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                worker = _SQLiteWorker(db_path)
                self._workers[db_path] = worker
            return worker


_SQLITE_WRITERS = _SQLiteWriterRegistry()


def log_to_sqlite(event_type: str, data: Dict[str, Any], db_path: str = SQLITE_PATH) -> str:
    """Store an event in the SQLite log."""

    _ensure_log_dir()
    event_id = generate_event_id(event_type)
    ts = datetime.utcnow().isoformat()
    router = GLOBAL_ROUTER or init_db_router("default")
    conn = router.get_connection("events", "write")
    payload = _AuditPayload(event_id, event_type, ts, dict(data))

    override_path = str(db_path) if db_path is not None else ""
    database_path = override_path or _connection_database_path(conn)
    if database_path in (":memory:", ""):
        _write_event_in_connection(conn, payload)
    else:
        _SQLITE_WRITERS.write(database_path, payload)

    return event_id


def get_recent_events(
    limit: int = 100,
    jsonl_path: str = JSONL_PATH,
    db_path: str = SQLITE_PATH,
) -> List[Dict[str, Any]]:
    """Return the most recent events from the JSONL or SQLite log."""
    if limit <= 0:
        return []
    router = GLOBAL_ROUTER or init_db_router("default")
    conn = router.get_connection("events")
    override_path = str(db_path) if db_path is not None else ""
    database_path = override_path or _connection_database_path(conn)

    target_conn: sqlite3.Connection
    if database_path in (":memory:", ""):
        target_conn = conn
    else:
        target_conn = sqlite3.connect(database_path)

    try:
        _ensure_db(target_conn)
        events: List[Dict[str, Any]] = []
        with closing(target_conn.cursor()) as cur:
            cur.execute(
                "SELECT event_id, timestamp, event_type FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            event_rows = cur.fetchall()
        for event_id, ts, etype in event_rows:
            with closing(target_conn.cursor()) as data_cur:
                data_cur.execute("SELECT key, value FROM event_data WHERE event_id=?", (event_id,))
                key_values = data_cur.fetchall()
            data = {k: v for k, v in key_values}
            try:
                for k, v in data.items():
                    if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                        data[k] = json.loads(v)
            except Exception:
                pass
            events.append({"timestamp": ts, "event_type": etype, "event_id": event_id, "data": data})
    finally:
        if target_conn is not conn:
            target_conn.close()
    if events:
        return list(reversed(events))
    if not Path(jsonl_path).exists():
        return []
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    lines.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return lines[:limit][::-1]


__all__ = [
    "generate_event_id",
    "log_event",
    "export_to_csv",
    "log_to_sqlite",
    "get_recent_events",
]
