"""Centralized, tamper-resistant audit logging for Security AI."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import queue
import random
import sqlite3
import threading
import time
import weakref
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING

from dynamic_path_router import resolve_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from db_router import DBRouter

# Directory and default log paths
LOG_DIR = resolve_dir("logs")
AUDIT_SQLITE_DIR = LOG_DIR / "audit"
JSONL_PATH = LOG_DIR / "audit_log.jsonl"
SQLITE_PATH = AUDIT_SQLITE_DIR / "audit_log.db"


logger = logging.getLogger(__name__)


GLOBAL_ROUTER: "DBRouter | None" = None


def _lazy_init_db_router(*args: Any, **kwargs: Any):
    """Import and proxy :func:`db_router.init_db_router` on first use."""

    module = importlib.import_module("db_router")
    init_fn = getattr(module, "init_db_router")
    global init_db_router
    init_db_router = init_fn  # type: ignore[assignment]
    return init_fn(*args, **kwargs)


init_db_router: Callable[..., "DBRouter"] = _lazy_init_db_router  # type: ignore[assignment]


def _get_db_router() -> "DBRouter":
    """Return a shared :class:`~db_router.DBRouter`, importing lazily."""

    global GLOBAL_ROUTER, init_db_router

    if GLOBAL_ROUTER is not None:
        return GLOBAL_ROUTER

    module = importlib.import_module("db_router")
    module_router = getattr(module, "GLOBAL_ROUTER", None)
    if module_router is not None:
        GLOBAL_ROUTER = module_router

    if init_db_router is _lazy_init_db_router:
        init_db_router = getattr(module, "init_db_router")  # type: ignore[assignment]

    if GLOBAL_ROUTER is None:
        GLOBAL_ROUTER = init_db_router("default")

    return GLOBAL_ROUTER


def _ensure_log_dir() -> None:
    """Create the log directory if missing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_SQLITE_DIR.mkdir(parents=True, exist_ok=True)


def generate_event_id(event_type: str) -> str:
    """Return a unique event identifier."""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"{event_type}-{ts}-{suffix}"


def _normalise_jsonl_path(jsonl_path: Path | str) -> Path:
    """Return the resolved :class:`Path` for *jsonl_path*."""

    path = Path(jsonl_path)
    if not path.is_absolute():
        path = LOG_DIR / path
    return path


def _append_record_to_jsonl(record: Dict[str, Any], jsonl_path: Path | str = JSONL_PATH) -> None:
    """Append *record* to the append-only JSONL audit log."""

    _ensure_log_dir()
    path = _normalise_jsonl_path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _build_record(event_type: str, data: Dict[str, Any], event_id: str | None = None) -> Dict[str, Any]:
    """Return the serialisable record for *event_type* and *data*."""

    record_id = event_id or generate_event_id(event_type)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "event_id": record_id,
        "data": data,
    }


def log_event(event_type: str, data: Dict[str, Any], jsonl_path: Path | str = JSONL_PATH) -> str:
    """Append an event record to the JSONL audit log."""

    record = _build_record(event_type, data)
    _append_record_to_jsonl(record, jsonl_path)
    return record["event_id"]


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
_SQLITE_BASE_DELAY = 0.1
_SQLITE_TIMEOUT_SECONDS = 30.0


@dataclass(slots=True)
class _AuditPayload:
    """Container for an audit record awaiting persistence."""

    event_id: str
    event_type: str
    timestamp: str
    data: Dict[str, Any]


@dataclass(slots=True)
class _QueuedAuditEvent:
    """Deferred audit payload waiting for an explicit flush."""

    payload: _AuditPayload
    db_path: str | Path


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


def _configure_sqlite_connection(conn: sqlite3.Connection) -> None:
    """Apply defensive PRAGMA settings to *conn* and close cursors eagerly."""

    # ``sqlite3.Connection.interrupt`` forcefully aborts any in-flight statements
    # associated with the connection.  In normal operation the newly-created
    # connections used by the audit mirror should not have pending work, but in
    # practice CPython may cache low-level SQLite handles between calls.  An
    # interrupt ensures we start from a clean slate even if a previous statement
    # was left unfinished by an interpreter-level quirk.
    try:
        conn.interrupt()
    except Exception:  # pragma: no cover - extremely defensive
        # ``interrupt`` was added in Python 3.6 and may raise if the underlying
        # connection has already been finalised.  Failing closed keeps the
        # configuration logic resilient on older runtimes while still giving us
        # best-effort protection against zombie statements.
        logger.debug("SQLite interrupt failed during configuration", exc_info=True)

    pragmas = (
        "PRAGMA busy_timeout=5000",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA journal_mode=WAL",
    )
    for pragma in pragmas:
        with closing(conn.execute(pragma)) as cur:
            # ``PRAGMA busy_timeout`` returns no rows while ``journal_mode`` does.
            # Fetching ensures the cursor is closed immediately, avoiding lingering
            # statements that can block subsequent commits.
            cur.fetchall()


def _resolve_sqlite_target(
    db_path: str | Path | None,
    operation: str,
) -> Tuple[str, sqlite3.Connection | None]:
    """Return the database path and optional router connection for *operation*."""

    if db_path is not None and str(db_path):
        return str(Path(db_path).expanduser()), None

    router = _get_db_router()
    conn = router.get_connection("events", operation)
    return _connection_database_path(conn), conn


def _write_event(db_path: str, payload: _AuditPayload) -> None:
    """Persist *payload* into *db_path* within an isolated connection."""

    thread_name = threading.current_thread().name
    logger.debug(
        "Opening fresh audit DB connection (event=%s, db=%s, thread=%s)",
        payload.event_id,
        db_path,
        thread_name,
    )
    if db_path not in (":memory:", ""):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(
        db_path,
        timeout=_SQLITE_TIMEOUT_SECONDS,
    ) as conn:
        _configure_sqlite_connection(conn)
        _ensure_db(conn)
        logger.debug(
            "Starting audit transaction (event=%s, thread=%s)",
            payload.event_id,
            thread_name,
        )
        try:
            with closing(conn.cursor()) as cur:
                cur.execute("BEGIN IMMEDIATE")
            logger.debug(
                "Creating audit cursor (event=%s, thread=%s)",
                payload.event_id,
                thread_name,
            )
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
            logger.debug(
                "Audit event queued for commit (event=%s, thread=%s)",
                payload.event_id,
                thread_name,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        else:
            logger.debug(
                "Committed audit event (event=%s, thread=%s)",
                payload.event_id,
                thread_name,
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
            if _should_retry_sqlite(exc) and attempt < _SQLITE_MAX_RETRIES - 1:
                delay = _SQLITE_BASE_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            raise
    if last_error is not None:
        raise last_error


_IN_MEMORY_LOCKS: "weakref.WeakKeyDictionary[sqlite3.Connection, threading.RLock]" = (
    weakref.WeakKeyDictionary()
)
_IN_MEMORY_LOCKS_GUARD = threading.Lock()


def _lock_for_connection(conn: sqlite3.Connection) -> threading.RLock:
    with _IN_MEMORY_LOCKS_GUARD:
        lock = _IN_MEMORY_LOCKS.get(conn)
        if lock is None:
            lock = threading.RLock()
            _IN_MEMORY_LOCKS[conn] = lock
        return lock


def _write_event_in_connection(conn: sqlite3.Connection, payload: _AuditPayload) -> None:
    """Persist *payload* using an existing in-memory connection."""

    lock = _lock_for_connection(conn)
    with lock:
        _configure_sqlite_connection(conn)
        _ensure_db(conn)
        thread_name = threading.current_thread().name
        logger.debug(
            "Writing audit event via shared connection (event=%s, thread=%s)",
            payload.event_id,
            thread_name,
        )
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
            logger.debug(
                "Audit event queued for commit on shared connection (event=%s, thread=%s)",
                payload.event_id,
                thread_name,
            )
            conn.commit()
            logger.debug(
                "Committed audit event on shared connection (event=%s, thread=%s)",
                payload.event_id,
                thread_name,
            )
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


_AUDIT_QUEUE_LOCK = threading.Lock()
_AUDIT_QUEUE: List[_QueuedAuditEvent] = []


def _mirror_payload_to_sqlite(
    db_path_spec: str | Path,
    payload: _AuditPayload,
) -> None:
    """Mirror *payload* into SQLite according to *db_path_spec*."""

    database_path, router_conn = _resolve_sqlite_target(db_path_spec, "write")
    if database_path in (":memory:", "") and router_conn is not None:
        _write_event_in_connection(router_conn, payload)
    elif database_path:
        _SQLITE_WRITERS.write(database_path, payload)


def log_to_sqlite(
    event_type: str,
    data: Dict[str, Any],
    db_path: str | Path = SQLITE_PATH,
    jsonl_path: Path | str = JSONL_PATH,
) -> str:
    """Store an event in the append-only log with a best-effort SQLite mirror."""

    record = _build_record(event_type, dict(data))
    _append_record_to_jsonl(record, jsonl_path)

    payload = _AuditPayload(
        record["event_id"], record["event_type"], record["timestamp"], dict(record["data"])
    )

    try:
        _mirror_payload_to_sqlite(db_path, payload)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to mirror audit event %s to SQLite (%s). Falling back to JSONL only.",
            payload.event_id,
            exc,
        )

    return record["event_id"]


def queue_event_for_later(
    event_type: str,
    data: Dict[str, Any],
    db_path: str | Path = SQLITE_PATH,
    jsonl_path: Path | str = JSONL_PATH,
) -> str:
    """Queue an audit event for deferred SQLite persistence.

    The event is written to the append-only JSONL log immediately while the SQLite
    mirror is postponed until :func:`flush_queued_events` is invoked.  This is
    useful when the SQLite database is temporarily locked by long-running read
    transactions.
    """

    record = _build_record(event_type, dict(data))
    _append_record_to_jsonl(record, jsonl_path)
    queued = _QueuedAuditEvent(
        payload=_AuditPayload(
            record["event_id"], record["event_type"], record["timestamp"], dict(record["data"])
        ),
        db_path=db_path,
    )
    with _AUDIT_QUEUE_LOCK:
        _AUDIT_QUEUE.append(queued)
    return record["event_id"]


def flush_queued_events() -> None:
    """Flush any deferred audit events to SQLite.

    Events that fail to persist remain queued and a :class:`RuntimeError` is
    raised to signal the partial failure.
    """

    with _AUDIT_QUEUE_LOCK:
        if not _AUDIT_QUEUE:
            return
        pending = list(_AUDIT_QUEUE)
        _AUDIT_QUEUE.clear()

    failures: List[_QueuedAuditEvent] = []
    for queued in pending:
        try:
            _mirror_payload_to_sqlite(queued.db_path, queued.payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to flush queued audit event %s: %s",
                queued.payload.event_id,
                exc,
            )
            failures.append(queued)

    if failures:
        with _AUDIT_QUEUE_LOCK:
            # Preserve ordering by re-inserting failed events at the front.
            _AUDIT_QUEUE[:0] = failures
        raise RuntimeError("Failed to flush one or more queued audit events")


def _read_events_from_jsonl(jsonl_path: Path | str, limit: int) -> List[Dict[str, Any]]:
    """Return at most *limit* events from the append-only JSONL log."""

    if limit <= 0:
        return []
    path = _normalise_jsonl_path(jsonl_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    lines.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return lines[:limit][::-1]


def get_recent_events(
    limit: int = 100,
    jsonl_path: Path | str = JSONL_PATH,
    db_path: str | Path = SQLITE_PATH,
) -> List[Dict[str, Any]]:
    """Return the most recent events preferring SQLite with JSONL fallback."""

    if limit <= 0:
        return []

    events: List[Dict[str, Any]] = []
    try:
        database_path, router_conn = _resolve_sqlite_target(db_path, "read")

        target_conn: sqlite3.Connection | None = router_conn
        if not (database_path in (":memory:", "") and router_conn is not None):
            target_conn = sqlite3.connect(database_path, timeout=_SQLITE_TIMEOUT_SECONDS)

        if target_conn is not None:
            _configure_sqlite_connection(target_conn)

        if target_conn is None:
            raise RuntimeError("Unable to resolve SQLite connection for audit events")

        try:
            _ensure_db(target_conn)
            with closing(target_conn.cursor()) as cur:
                cur.execute(
                    "SELECT event_id, timestamp, event_type FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                event_rows = cur.fetchall()
            for event_id, ts, etype in event_rows:
                with closing(target_conn.cursor()) as data_cur:
                    data_cur.execute(
                        "SELECT key, value FROM event_data WHERE event_id=?", (event_id,)
                    )
                    key_values = data_cur.fetchall()
                data = {k: v for k, v in key_values}
                try:
                    for k, v in data.items():
                        if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                            data[k] = json.loads(v)
                except Exception:
                    pass
                events.append(
                    {"timestamp": ts, "event_type": etype, "event_id": event_id, "data": data}
                )
        finally:
            if target_conn is not None and target_conn is not router_conn:
                target_conn.close()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to read audit events from SQLite (%s). Falling back to JSONL only.",
            exc,
        )
        events = []

    if events:
        return list(reversed(events))
    return _read_events_from_jsonl(jsonl_path, limit)


__all__ = [
    "generate_event_id",
    "log_event",
    "export_to_csv",
    "log_to_sqlite",
    "queue_event_for_later",
    "flush_queued_events",
    "get_recent_events",
]
