"""Database routing utilities for Menace.

This module exposes a :class:`DBRouter` that decides whether a table should
reside in the local or the shared SQLite database.  Shared tables are
available to every Menace instance while local tables are isolated per
``menace_id``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Set

__all__ = [
    "DBRouter",
    "SHARED_TABLES",
    "LOCAL_TABLES",
    "DENY_TABLES",
    "init_db_router",
    "GLOBAL_ROUTER",
]


# Tables stored in the shared database.  These tables are visible to every
# Menace instance.  The container is mutated in-place on reload so existing
# references (e.g. in tests) observe the updated contents.
if "SHARED_TABLES" in globals():
    SHARED_TABLES.clear()
    SHARED_TABLES.update(
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
if "LOCAL_TABLES" in globals():
    LOCAL_TABLES.clear()
    LOCAL_TABLES.update(
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
            "roi_events",
            "action_roi",
            "allocation_weights",
            "ledger",
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
            "failures",
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
        "roi_events",
        "action_roi",
        "allocation_weights",
        "ledger",
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
        "failures",
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

# Tables explicitly denied even if present in the allow lists.  Also mutated
# in-place on reload.
if "DENY_TABLES" in globals():
    DENY_TABLES.clear()
    DENY_TABLES.update({"capital_ledger", "finance_logs"})
else:
    DENY_TABLES: Set[str] = {"capital_ledger", "finance_logs"}


def _load_table_overrides() -> None:
    """Extend allow/deny lists from env vars or optional config file.

    Environment variables ``DB_ROUTER_SHARED_TABLES``, ``DB_ROUTER_LOCAL_TABLES``
    and ``DB_ROUTER_DENY_TABLES`` accept comma separated table names.  A JSON
    config file referenced via ``DB_ROUTER_CONFIG`` may define ``shared``,
    ``local`` and ``deny`` arrays.
    """

    global _audit_log_path

    shared_env = os.getenv("DB_ROUTER_SHARED_TABLES", "")
    local_env = os.getenv("DB_ROUTER_LOCAL_TABLES", "")
    deny_env = os.getenv("DB_ROUTER_DENY_TABLES", "")
    config_path = os.getenv("DB_ROUTER_CONFIG")
    if not config_path:
        default_cfg = Path(__file__).resolve().parent / "config" / "db_router_tables.json"
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
            if not _audit_log_path:
                _audit_log_path = data.get("audit_log")
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
# the referenced file as JSON lines.
_audit_log_path: str | None = os.getenv("DB_ROUTER_AUDIT_LOG")
_load_table_overrides()


def _record_audit(entry: dict[str, str]) -> None:
    """Persist *entry* to the audit log when configured."""

    if not _audit_log_path:
        return
    try:
        directory = os.path.dirname(_audit_log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(_audit_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # pragma: no cover - best effort
        pass


# Global router instance used by modules that rely on a single router without
# passing it around explicitly.  ``init_db_router`` populates this value.
GLOBAL_ROUTER: "DBRouter" | None = None


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
            if os.path.isdir(local_db_path)
            else local_db_path
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.local_conn = sqlite3.connect(local_path, check_same_thread=False)

        os.makedirs(os.path.dirname(shared_db_path), exist_ok=True)
        self.shared_conn = sqlite3.connect(shared_db_path, check_same_thread=False)

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
        accesses are silent except for the optional audit log.
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

            conn: sqlite3.Connection
            if table_name in SHARED_TABLES:
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
    defaults to ``./menace_<id>_local.db`` and ``shared_db_path`` defaults to
    ``./shared/global.db`` when not provided. The created router is stored in
    :data:`GLOBAL_ROUTER` and returned.
    """

    global GLOBAL_ROUTER

    local_path = local_db_path or f"./menace_{menace_id}_local.db"
    shared_path = shared_db_path or "./shared/global.db"

    GLOBAL_ROUTER = DBRouter(menace_id, local_path, shared_path)
    return GLOBAL_ROUTER


