"""Database routing utilities for Menace.

This module exposes a :class:`DBRouter` that decides whether a table should
reside in the local or the shared SQLite database.  Shared tables are
available to every Menace instance while local tables are isolated per
``menace_id``.  Connections returned by :func:`DBRouter.get_connection` are
wrapped so that any ``execute`` call automatically records the number of rows
read or written via :func:`audit.log_db_access`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
try:  # pragma: no cover - optional dependency
    import sqlparse
    from sqlparse.sql import Identifier, IdentifierList, Parenthesis
    from sqlparse.tokens import Keyword
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully
    sqlparse = None  # type: ignore

    class _SQLParseStub:  # pragma: no cover - lightweight placeholder
        pass

    Identifier = IdentifierList = Parenthesis = _SQLParseStub  # type: ignore
    Keyword = object()
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Set, Iterable, Mapping, Any

from audit import log_db_access
from dynamic_path_router import get_project_root, resolve_path


__all__ = [
    "DBRouter",
    "SHARED_TABLES",
    "LOCAL_TABLES",
    "DENY_TABLES",
    "init_db_router",
    "GLOBAL_ROUTER",
    "queue_insert",
    "configure_db_router_audit_logging",
]


# Tables stored in the shared database.  These tables are visible to every
# Menace instance.  The container is mutated in-place on reload so existing
# references (e.g. in tests) observe the updated contents.
if globals().get("SHARED_TABLES") is not None:
    SHARED_TABLES.clear()  # noqa: F821
    SHARED_TABLES.update(  # noqa: F821
        {
            "bots",
            "code",
            "discrepancies",
            "enhancements",
            "errors",
            "information",
            "overrides",
            "workflow_summaries",
            "telemetry",
        }
    )
else:
    SHARED_TABLES: Set[str] = {
        "bots",
        "code",
        "discrepancies",
        "enhancements",
        "errors",
        "information",
        "overrides",
        "workflow_summaries",
        "telemetry",
    }

# Tables stored in the local database.  These are private to a specific
# ``menace_id`` instance.  The list is mutated in-place on reload to keep
# references stable.
if globals().get("LOCAL_TABLES") is not None:
    LOCAL_TABLES.clear()  # noqa: F821
    LOCAL_TABLES.update(  # noqa: F821
        {
            "events",
            "memory",
            "menace_config",
            "models",
            "patch_history",
            "roi_logs",
            "sandbox_metrics",
            "variants",
            "metrics",
            "revenue",
            "subs",
            "churn",
            "leads",
            "profit",
            "history",
            "patches",
            "healing_actions",
            "tasks",
            "metadata",
            "vector_metrics",
            "roi_telemetry",
            "roi_prediction_events",
            "results",
            "resolutions",
            "deployments",
            "bot_trials",
            "update_history",
            "updates",
            "roi_events",
            "action_roi",
            "allocation_weights",
            "ledger",
            "stripe_ledger",
            "profit_history",
            "capital_summary",
            "bot_performance",
            "maintenance",
            "allocations",
            "risk_summaries",
            "evolutions",
            "evolution_history",
            "retrieval_stats",
            "retriever_stats",
            "retrieval_cache",
            "experiment_history",
            "experiment_tests",
            "embedding_stats",
            "messages",
            "evaluation",
            "evaluation_history",
            "weight_override",
            "roi",
            "detections",
            "failures",
            "anomalies",
            "synergy_history",
            "feedback",
            "mirror_logs",
            "efficiency",
            "saturation",
            "action_audit",
            "db_roi_metrics",
            "decisions",
            "flakiness_history",
            "investments",
            "module_metrics",
            "patch_outcomes",
            "patch_provenance",
            "retriever_kpi",
            "roi_history",
            "test_history",
            "workflows",
        }
    )
else:
    LOCAL_TABLES: Set[str] = {
        "events",
        "memory",
        "menace_config",
        "models",
        "patch_history",
        "roi_logs",
        "sandbox_metrics",
        "variants",
        "metrics",
        "revenue",
        "subs",
        "churn",
        "leads",
        "profit",
        "history",
        "patches",
        "healing_actions",
        "tasks",
        "metadata",
        "vector_metrics",
        "roi_telemetry",
        "roi_prediction_events",
        "results",
        "resolutions",
        "deployments",
        "bot_trials",
        "update_history",
        "updates",
        "roi_events",
        "action_roi",
        "allocation_weights",
        "ledger",
        "stripe_ledger",
        "profit_history",
        "capital_summary",
        "bot_performance",
        "maintenance",
        "allocations",
        "risk_summaries",
        "evolutions",
        "evolution_history",
        "retrieval_stats",
        "retriever_stats",
        "retrieval_cache",
        "experiment_history",
        "experiment_tests",
        "embedding_stats",
        "messages",
        "evaluation",
        "evaluation_history",
        "weight_override",
        "roi",
        "detections",
        "failures",
        "anomalies",
        "synergy_history",
        "feedback",
        "mirror_logs",
        "efficiency",
        "saturation",
        "action_audit",
        "db_roi_metrics",
        "decisions",
        "flakiness_history",
        "investments",
        "module_metrics",
        "patch_outcomes",
        "patch_provenance",
        "retriever_kpi",
        "roi_history",
        "test_history",
        "workflows",
    }

# Ensure evaluation results are stored locally
LOCAL_TABLES.add("evaluation")

# Tables explicitly denied even if present in the allow lists.  Also mutated
# in-place on reload.
if globals().get("DENY_TABLES") is not None:
    DENY_TABLES.clear()  # noqa: F821
    DENY_TABLES.update({"capital_ledger", "finance_logs"})  # noqa: F821
else:
    DENY_TABLES: Set[str] = {"capital_ledger", "finance_logs"}


def _load_table_overrides() -> None:
    """Extend allow/deny lists from env vars or optional config file.

    Environment variables ``DB_ROUTER_SHARED_TABLES``, ``DB_ROUTER_LOCAL_TABLES``
    and ``DB_ROUTER_DENY_TABLES`` accept comma separated table names.  A JSON
    config file referenced via ``DB_ROUTER_CONFIG`` may define ``shared``,
    ``local`` and ``deny`` arrays.
    """

    global _audit_log_candidate

    shared_env = os.getenv("DB_ROUTER_SHARED_TABLES", "")
    local_env = os.getenv("DB_ROUTER_LOCAL_TABLES", "")
    deny_env = os.getenv("DB_ROUTER_DENY_TABLES", "")
    config_path = os.getenv("DB_ROUTER_CONFIG")
    if not config_path:
        default_cfg = resolve_path("config/db_router_tables.json")
        if default_cfg.exists():
            config_path = str(default_cfg)

    def _split(value: str) -> Set[str]:
        return {t.strip() for t in value.split(",") if t.strip()}

    shared_extra = _split(shared_env)
    local_extra = _split(local_env)
    deny_extra = _split(deny_env)

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            shared_extra.update(data.get("shared", []))
            local_extra.update(data.get("local", []))
            deny_extra.update(data.get("deny", []))
            if not _audit_log_candidate:
                _audit_log_candidate = data.get("audit_log")
        except Exception:
            # Ignore malformed config files; routing will fall back to defaults.
            pass

    SHARED_TABLES.update(shared_extra)
    LOCAL_TABLES.update(local_extra)
    DENY_TABLES.update(deny_extra)
    for table in deny_extra:
        SHARED_TABLES.discard(table)
        LOCAL_TABLES.discard(table)


logger = logging.getLogger(__name__)
_level_name = os.getenv("DB_ROUTER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _level_name, logging.INFO))
_log_format = os.getenv("DB_ROUTER_LOG_FORMAT", "json").lower()

# Optional audit log for table accesses. When ``DB_ROUTER_AUDIT_LOG`` is defined
# or ``audit_log`` is provided in the DB router config, entries are appended to
# the referenced file as JSON lines once audit logging has been initialised via
# :func:`configure_db_router_audit_logging`.
_audit_log_candidate: str | None = os.getenv("DB_ROUTER_AUDIT_LOG")
_audit_log_path: str | None = None
_audit_logging_enabled = False
_load_table_overrides()


def _record_audit(entry: dict[str, str]) -> None:
    """Persist *entry* to the audit log when configured."""

    if not _audit_logging_enabled or not _audit_log_path:
        return
    try:
        directory = os.path.dirname(_audit_log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(_audit_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # pragma: no cover - best effort
        pass


def configure_db_router_audit_logging(*, audit_log_path: str | None = None) -> None:
    """Enable audit logging for DB router activity.

    The audit log path may be provided explicitly via ``audit_log_path`` or is
    resolved from environment/configuration captured during module import.  No
    directories are created and no files are written until the first audit
    record is produced.
    """

    global _audit_log_path, _audit_logging_enabled

    if audit_log_path is None:
        audit_log_path = _audit_log_candidate

    _audit_log_path = audit_log_path
    _audit_logging_enabled = bool(_audit_log_path)


# Global router instance used by modules that rely on a single router without
# passing it around explicitly.  ``init_db_router`` populates this value.
GLOBAL_ROUTER: "DBRouter" | None = None


def queue_insert(table: str, record: dict[str, Any], menace_id: str) -> None:
    """Queue an ``INSERT`` operation for *table*.

    For tables listed in :data:`SHARED_TABLES` the record is appended to a
    menace-specific queue file via :func:`db_write_queue.append_record`.  Local
    table writes are applied synchronously using :data:`GLOBAL_ROUTER`.

    The base directory for queue files defaults to the location used by
    :mod:`db_write_queue` but may be overridden via the
    ``DB_ROUTER_QUEUE_DIR`` environment variable.
    """

    from db_write_queue import append_record

    if table in DENY_TABLES:
        raise ValueError(f"Access to table '{table}' is denied")

    queue_dir_env = os.getenv("DB_ROUTER_QUEUE_DIR")
    queue_dir = Path(queue_dir_env) if queue_dir_env else None

    if table in SHARED_TABLES:
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "menace_id": menace_id,
            "table_name": table,
            "operation": "queue_insert",
            "timestamp": timestamp,
        }
        if _log_format == "kv":
            msg = " ".join(f"{k}={v}" for k, v in entry.items())
        else:
            msg = json.dumps(entry)
        logger.info(msg)

        append_record(table, record, menace_id, queue_dir=queue_dir)
        conn = GLOBAL_ROUTER.shared_conn if GLOBAL_ROUTER else None
        log_db_access("write", table, 1, menace_id, db_conn=conn)

        _record_audit(entry)
    elif table in LOCAL_TABLES:
        if GLOBAL_ROUTER is None:
            raise RuntimeError("GLOBAL_ROUTER is not initialised")
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        GLOBAL_ROUTER.execute_and_log(
            table,
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(record.values()),
        )
        _record_audit(
            {
                "menace_id": menace_id,
                "table_name": table,
                "operation": "insert",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    else:
        raise ValueError(f"Unknown table: {table}")


class LoggedCursor(sqlite3.Cursor):
    """Cursor that records database access via :func:`log_db_access`."""

    menace_id: str
    _rows: list[Any] | None = None

    def _table_from_sql(self, sql: str) -> str:
        if sqlparse is None:  # pragma: no cover - fallback without sqlparse
            return "unknown"
        try:
            statement = sqlparse.parse(sql)[0]
        except Exception:
            return "unknown"

        def extract_identifier(token):
            if isinstance(token, Identifier):
                real = token.get_real_name()
                if real and not (
                    token.is_group
                    and token.tokens
                    and isinstance(token.tokens[0], Parenthesis)
                ):
                    return real
                # Handle sub-selects or identifiers containing groups
                for sub in token.tokens:
                    name = extract_identifier(sub)
                    if name:
                        return name
                return real or token.get_name()
            if isinstance(token, IdentifierList):
                first = next(token.get_identifiers(), None)
                if first:
                    return extract_identifier(first)
            if token.is_group:
                for sub in token.tokens:
                    name = extract_identifier(sub)
                    if name:
                        return name
            return None

        stmt_type = statement.get_type()
        if stmt_type == "UPDATE":
            tokens = [t for t in statement.tokens if not t.is_whitespace]
            if len(tokens) > 1:
                name = extract_identifier(tokens[1])
                return name.strip('"`[]') if name else "unknown"
            return "unknown"

        keyword = {"SELECT": "FROM", "DELETE": "FROM", "INSERT": "INTO"}.get(
            stmt_type
        )
        if not keyword:
            return "unknown"

        from_seen = False
        for token in statement.tokens:
            if token.is_whitespace:
                continue
            if from_seen:
                name = extract_identifier(token)
                if name:
                    return name.strip('"`[]')
            elif token.ttype is Keyword and token.value.upper() == keyword:
                from_seen = True
        return "unknown"

    def _log(self, action: str, table: str, row_count: int) -> None:
        log_db_access(
            action,
            table,
            row_count,
            self.menace_id,
            db_conn=self.connection,
        )

    def execute(self, sql: str, parameters: Iterable | None = None):  # type: ignore[override]
        super().execute(sql, parameters or ())
        table = self._table_from_sql(sql)
        is_read = sql.lstrip().upper().startswith("SELECT")
        if is_read:
            self._rows = super().fetchall()
            row_count = len(self._rows)
        else:
            self._rows = None
            row_count = self.rowcount if self.rowcount != -1 else 0
        self._log("read" if is_read else "write", table, row_count)
        return self

    def executemany(
        self, sql: str, seq_of_parameters: Iterable[Iterable]
    ):  # type: ignore[override]
        super().executemany(sql, seq_of_parameters)
        table = self._table_from_sql(sql)
        is_read = sql.lstrip().upper().startswith("SELECT")
        if is_read:
            self._rows = super().fetchall()
            row_count = len(self._rows)
        else:
            self._rows = None
            row_count = self.rowcount if self.rowcount != -1 else 0
        self._log("read" if is_read else "write", table, row_count)
        return self

    def fetchone(self):  # type: ignore[override]
        if self._rows is not None:
            return self._rows.pop(0) if self._rows else None
        return super().fetchone()

    def fetchall(self):  # type: ignore[override]
        if self._rows is not None:
            rows, self._rows = self._rows, []
            return rows
        return super().fetchall()

    def fetchmany(self, size: int | None = None):  # type: ignore[override]
        if self._rows is not None:
            if size is None:
                size = 1
            rows = self._rows[:size]
            self._rows = self._rows[size:]
            return rows
        return super().fetchmany(size)

    def __iter__(self):  # type: ignore[override]
        if self._rows is not None:
            while self._rows:
                yield self._rows.pop(0)
        else:
            yield from super().__iter__()


def _configure_sqlite_connection(conn: sqlite3.Connection) -> None:
    """Apply pragmatic settings that reduce writer contention."""

    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # Some SQLite builds (e.g. older Android) do not support WAL mode; ignore.
        pass
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")


class LoggedConnection(sqlite3.Connection):
    """Connection whose cursors automatically log row counts."""

    menace_id: str

    def cursor(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("factory", LoggedCursor)
        cur: LoggedCursor = super().cursor(*args, **kwargs)  # type: ignore[assignment]
        cur.menace_id = self.menace_id
        return cur

    def execute(self, *args, **kwargs):  # type: ignore[override]
        return self.cursor().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):  # type: ignore[override]
        return self.cursor().executemany(*args, **kwargs)


class DBRouter:
    """Route table operations to local or shared SQLite databases."""

    def __init__(self, menace_id: str, local_db_path: str, shared_db_path: str) -> None:
        """Create a new :class:`DBRouter`.

        Parameters
        ----------
        menace_id:
            Identifier for the Menace instance.  When ``local_db_path`` points to
            a directory a database file named ``"<menace_id>.db"`` will be created
            inside that directory.
        local_db_path:
            Path to the SQLite database used for local tables or a directory in
            which a database for ``menace_id`` should be created.
        shared_db_path:
            Path to the SQLite database used for shared tables.
        """

        self.menace_id = menace_id
        # When ``local_db_path`` is a directory, create a database file for this
        # menace instance inside it.
        local_path = (
            os.path.join(local_db_path, f"{menace_id}.db")
            if local_db_path != ":memory:" and os.path.isdir(local_db_path)
            else local_db_path
        )
        if local_path != ":memory:":
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.local_conn: LoggedConnection = sqlite3.connect(  # noqa: SQL001
            local_path, check_same_thread=False, factory=LoggedConnection
        )  # type: ignore[assignment]
        self.local_conn.menace_id = menace_id
        _configure_sqlite_connection(self.local_conn)

        if shared_db_path != ":memory:":
            os.makedirs(os.path.dirname(shared_db_path), exist_ok=True)
        self.shared_conn: LoggedConnection = sqlite3.connect(  # noqa: SQL001
            shared_db_path, check_same_thread=False, factory=LoggedConnection
        )  # type: ignore[assignment]
        self.shared_conn.menace_id = menace_id
        _configure_sqlite_connection(self.shared_conn)

        # ``threading.Lock`` protects against concurrent access when deciding
        # which connection to return.
        self._lock = threading.Lock()
        self._access_counts = {
            "shared": defaultdict(int),
            "local": defaultdict(int),
        }

        # Background reporting thread placeholders
        self._report_thread: threading.Thread | None = None
        self._report_stop: threading.Event | None = None

        interval = os.getenv("DB_ROUTER_METRICS_INTERVAL")
        if interval:
            try:
                seconds = float(interval)
            except ValueError:
                seconds = 0.0
            if seconds > 0:
                self.start_periodic_reporting(seconds)

    # ------------------------------------------------------------------
    def __enter__(self) -> "DBRouter":
        """Return ``self`` when entering a context manager."""

        return self

    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc, tb) -> None:
        """Close connections when leaving a context manager."""

        self.close()

    # ------------------------------------------------------------------
    def get_connection(self, table_name: str, operation: str = "read") -> sqlite3.Connection:
        """Return the appropriate connection for ``table_name``.

        Parameters
        ----------
        table_name:
            Name of the table being accessed.
        operation:
            Type of operation being performed (for example ``"read"`` or
            ``"write"``).  The value is recorded in the audit log and metrics.

        A :class:`ValueError` is raised for unknown tables.  Shared table
        accesses emit a structured log entry for observability while local table
        accesses are silent except for the optional audit log.  The returned
        :class:`LoggedConnection` automatically records the number of rows read
        or written for each executed statement.
        """

        if not table_name:
            raise ValueError("table_name must be a non-empty string")

        with self._lock:
            if table_name in DENY_TABLES:
                raise ValueError(f"Access to table '{table_name}' is denied")

            timestamp = datetime.utcnow().isoformat()
            entry = {
                "menace_id": self.menace_id,
                "table_name": table_name,
                "operation": operation,
                "timestamp": timestamp,
            }

            conn: LoggedConnection
            if table_name in SHARED_TABLES:
                if operation == "write":
                    logger.warning(
                        "direct write to shared table '%s'; use queue_write instead",
                        table_name,
                    )
                if _log_format == "kv":
                    msg = " ".join(f"{k}={v}" for k, v in entry.items())
                else:
                    msg = json.dumps(entry)
                logger.info(msg)
                self._access_counts["shared"][table_name] += 1
                conn = self.shared_conn
            elif table_name in LOCAL_TABLES:
                self._access_counts["local"][table_name] += 1
                conn = self.local_conn
            else:
                raise ValueError(f"Unknown table: {table_name}")

            _record_audit(entry)

            return conn

    # ------------------------------------------------------------------
    def execute_and_log(
        self,
        table_name: str,
        sql: str,
        parameters: Iterable | Mapping | None = None,
    ):
        """Execute *sql* against *table_name*.

        The returned :class:`LoggedConnection` handles auditing automatically.
        For ``SELECT`` statements the fetched rows are returned; for writes the
        cursor is returned after committing the transaction.

        Parameters
        ----------
        table_name:
            Name of the table being accessed.
        sql:
            SQL statement to execute.
        parameters:
            Parameters for the SQL statement.
        """

        is_read = sql.lstrip().upper().startswith("SELECT")
        conn = self.get_connection(table_name, "read" if is_read else "write")
        cursor = conn.execute(sql, parameters or ())
        if is_read:
            return cursor.fetchall()

        conn.commit()
        return cursor

    # ------------------------------------------------------------------
    def queue_write(
        self,
        table_name: str,
        values: Mapping[str, Any],
        hash_fields: Iterable[str],
    ) -> None:
        """Queue a write to a shared table.

        Parameters
        ----------
        table_name:
            Target table name. Must be one of :data:`SHARED_TABLES`.
        values:
            Mapping of column names to values for the row.
        hash_fields:
            Iterable of field names used to compute a deduplication hash.
        """

        if table_name not in SHARED_TABLES:
            raise ValueError("queue_write is only supported for shared tables")
        from db_write_queue import append_record, queue_insert as queue_insert_record
        append_record(table_name, values, self.menace_id)
        queue_insert_record(table_name, values, hash_fields)
        log_db_access(
            "write",
            table_name,
            1,
            self.menace_id,
            db_conn=self.shared_conn,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close both the local and shared database connections."""
        self.stop_periodic_reporting()
        self.local_conn.close()
        self.shared_conn.close()

    # ------------------------------------------------------------------
    def start_periodic_reporting(self, interval: float = 60.0) -> None:
        """Start a background thread periodically flushing metrics.

        Parameters
        ----------
        interval:
            Seconds between metric flushes.
        """

        if self._report_thread and self._report_thread.is_alive():
            return
        stop = threading.Event()
        self._report_stop = stop

        def _worker() -> None:
            while not stop.wait(interval):
                try:
                    self.get_access_counts(flush=True)
                except Exception:  # pragma: no cover - best effort
                    pass

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        self._report_thread = thread

    # ------------------------------------------------------------------
    def stop_periodic_reporting(self) -> None:
        """Stop the background reporting thread if active."""

        if self._report_stop is not None:
            self._report_stop.set()
        thread = self._report_thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._report_thread = None
        self._report_stop = None

    # ------------------------------------------------------------------
    def get_access_counts(self, *, flush: bool = False) -> dict[str, dict[str, int]]:
        """Return a snapshot of table access counts for monitoring.

        When ``flush`` is true the counts are also forwarded to the telemetry
        backend via ``telemetry_backend.record_table_access`` and then reset.
        """

        with self._lock:
            snapshot = {
                kind: dict(counts) for kind, counts in self._access_counts.items()
            }

            if flush:
                _tb = None
                try:  # pragma: no cover - import available only in package context
                    from . import telemetry_backend as _tb  # type: ignore
                except Exception:  # pragma: no cover - fallback for tests
                    try:
                        import telemetry_backend as _tb  # type: ignore
                    except Exception:
                        _tb = None
                if _tb is not None:
                    for kind, counts in snapshot.items():
                        for table, count in counts.items():
                            try:
                                _tb.record_table_access(
                                    self.menace_id, table, kind, count
                                )
                            except Exception:  # pragma: no cover - best effort
                                pass
                for counts in self._access_counts.values():
                    counts.clear()

            return snapshot


def init_db_router(
    menace_id: str,
    local_db_path: str | None = None,
    shared_db_path: str | None = None,
) -> DBRouter:
    """Initialise a global :class:`DBRouter` instance.

    Entry points must invoke this before performing any database operations so
    that :data:`GLOBAL_ROUTER` is available to imported modules. ``local_db_path``
    defaults to ``menace_<id>_local.db`` and ``shared_db_path`` defaults to
    ``shared/global.db`` when not provided; both are resolved relative to the
    repository root via :func:`resolve_path`. The created router is stored in
    :data:`GLOBAL_ROUTER` and returned.
    """

    global GLOBAL_ROUTER

    project_root = get_project_root()

    if local_db_path == ":memory:":
        local_path_str = ":memory:"
    elif local_db_path is None:
        try:
            local_path = resolve_path(f"menace_{menace_id}_local.db")
        except FileNotFoundError:
            local_path = (project_root / f"menace_{menace_id}_local.db").resolve()
        else:
            local_path = local_path.resolve()
        local_path_str = str(local_path)
    else:
        local_path = Path(local_db_path).expanduser().resolve()
        local_path_str = str(local_path)

    if shared_db_path == ":memory:":
        shared_path_str = ":memory:"
    elif shared_db_path is None:
        try:
            shared_path = resolve_path("shared/global.db")
        except FileNotFoundError:
            shared_path = (project_root / "shared" / "global.db").resolve()
        else:
            shared_path = shared_path.resolve()
        shared_path_str = str(shared_path)
    else:
        shared_path = Path(shared_db_path).expanduser().resolve()
        shared_path_str = str(shared_path)

    configure_db_router_audit_logging()

    GLOBAL_ROUTER = DBRouter(menace_id, local_path_str, shared_path_str)
    return GLOBAL_ROUTER
