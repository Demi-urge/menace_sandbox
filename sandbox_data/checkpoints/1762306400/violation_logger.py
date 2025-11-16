"""Append-only violation logging utilities for Security AI."""

from __future__ import annotations

import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock
from typing import Any, List, Dict, Optional

from dynamic_path_router import get_project_root, resolve_dir
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

if __package__:
    from .retry_utils import publish_with_retry
else:  # pragma: no cover - fallback for direct execution
    from retry_utils import publish_with_retry  # type: ignore
import db_router
import sqlite3
from sandbox_settings import SandboxSettings
from logging_utils import (
    LockedRotatingFileHandler,
    LockedTimedRotatingFileHandler,
)

try:  # optional dependency
    from .unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - bus optional
    UnifiedEventBus = None  # type: ignore

# Directory and file path for violation logs
try:
    LOG_DIR = Path(resolve_dir("logs"))
except (FileNotFoundError, NotADirectoryError):
    LOG_DIR = (get_project_root() / "logs").resolve()

LOG_PATH: Path | str = LOG_DIR / "violation_log.jsonl"
DEFAULT_ALIGNMENT_DB_PATH = LOG_DIR / "alignment_warnings.db"
ALIGNMENT_DB_PATH: Path | str = DEFAULT_ALIGNMENT_DB_PATH

logger = logging.getLogger(__name__)
_logger_lock = Lock()
_loggers: Dict[str, logging.Logger] = {}
_alignment_conn_cache: sqlite3.Connection | None = None
_alignment_conn_path: Path | None = None
_alignment_conn_lock = Lock()

_event_bus: Optional[UnifiedEventBus]
if UnifiedEventBus is not None:
    try:
        _event_bus = UnifiedEventBus()
    except Exception:  # pragma: no cover - runtime optional
        _event_bus = None
else:  # pragma: no cover - bus not available
    _event_bus = None


def set_event_bus(bus: Optional[UnifiedEventBus]) -> None:
    """Override the global event bus instance."""
    global _event_bus
    _event_bus = bus


def _ensure_log_dir() -> None:
    """Create the log directory if it doesn't exist."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def _get_logger(path: Path | str) -> logging.Logger:
    path_obj = Path(path)
    path_str = os.fspath(path_obj)
    with _logger_lock:
        lg = _loggers.get(path_str)
        if lg is None:
            settings = SandboxSettings()
            lg = logging.getLogger(f"violation_logger_{path_str}")
            lg.setLevel(logging.INFO)
            rotate_seconds = settings.log_rotation_seconds
            handler = _build_file_handler(path_str, settings, rotate_seconds)
            handler.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(handler)
            lg.propagate = False
            _loggers[path_str] = lg
        return lg


def _build_file_handler(
    path_str: str, settings: SandboxSettings, rotate_seconds: int | None
) -> logging.Handler:
    """Return a file handler, falling back when ``filelock`` is unavailable."""

    if rotate_seconds:
        handler_cls = LockedTimedRotatingFileHandler
        fallback_cls = TimedRotatingFileHandler
        kwargs = dict(
            when="s",
            interval=rotate_seconds,
            backupCount=settings.log_rotation_backup_count,
            encoding="utf-8",
        )
    else:
        handler_cls = LockedRotatingFileHandler
        fallback_cls = RotatingFileHandler
        kwargs = dict(
            maxBytes=settings.log_rotation_max_bytes,
            backupCount=settings.log_rotation_backup_count,
            encoding="utf-8",
        )

    try:
        return handler_cls(path_str, **kwargs)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        if exc.name != "filelock":
            raise
        logger.warning(
            "filelock dependency missing; falling back to unlocked file handler"
        )
        return fallback_cls(path_str, **kwargs)


def _global_router() -> "db_router.DBRouter | None":
    """Return the live global router reference if initialised."""

    return getattr(db_router, "GLOBAL_ROUTER", None)


def _use_router_for_alignment(
    alignment_db_path: Path, router: "db_router.DBRouter | None" = None
) -> bool:
    """Return ``True`` when the global router should service alignment logs."""

    if router is None:
        router = _global_router()
    return (
        router is not None
        and alignment_db_path.resolve() == Path(DEFAULT_ALIGNMENT_DB_PATH).resolve()
    )


def _alignment_conn() -> sqlite3.Connection:
    """Return a connection for alignment warnings."""
    global _alignment_conn_cache, _alignment_conn_path

    alignment_db_path = Path(ALIGNMENT_DB_PATH)

    router = _global_router()
    if router is not None and _use_router_for_alignment(alignment_db_path, router):
        return router.get_connection("errors")

    alignment_db_path.parent.mkdir(parents=True, exist_ok=True)
    with _alignment_conn_lock:
        if _alignment_conn_cache is None or _alignment_conn_path != alignment_db_path:
            if _alignment_conn_cache is not None:
                try:
                    _alignment_conn_cache.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            _alignment_conn_cache = sqlite3.connect(
                alignment_db_path, check_same_thread=False
            )
            _alignment_conn_path = alignment_db_path
        return _alignment_conn_cache


def persist_alignment_warning(record: Dict[str, Any]) -> None:
    """Persist a single alignment warning to the dedicated SQLite store."""

    _ensure_log_dir()
    conn = _alignment_conn()
    patch_link = None
    evidence = record.get("evidence") or {}
    if isinstance(evidence, dict):
        patch_link = evidence.get("patch_link") or evidence.get("patch_id")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warnings (
            timestamp INTEGER,
            entry_id TEXT,
            violation_type TEXT,
            severity INTEGER,
            patch_link TEXT,
            review_status TEXT DEFAULT 'pending'
        )
        """
    )
    # Ensure the new column exists for pre-existing databases
    cols = [r[1] for r in conn.execute("PRAGMA table_info(warnings)").fetchall()]
    if "review_status" not in cols:
        conn.execute(
            "ALTER TABLE warnings ADD COLUMN review_status TEXT DEFAULT 'pending'"
        )
        conn.execute(
            "UPDATE warnings SET review_status = 'pending' WHERE review_status IS NULL"
        )
    conn.execute(
        "INSERT INTO warnings (timestamp, entry_id, violation_type, severity, patch_link, review_status)"
        " VALUES (?, ?, ?, ?, ?, 'pending')",
        (
            record.get("timestamp"),
            record.get("entry_id"),
            record.get("violation_type"),
            record.get("severity"),
            patch_link,
        ),
    )
    conn.commit()


def load_persisted_alignment_warnings(
    limit: int = 50,
    min_severity: int | None = None,
    max_severity: int | None = None,
    review_status: str | None = None,
) -> List[Dict[str, Any]]:
    """Load alignment warnings from the SQLite store."""

    if limit <= 0:
        return []
    alignment_db_path = Path(ALIGNMENT_DB_PATH)
    if (not _use_router_for_alignment(alignment_db_path)) and not alignment_db_path.exists():
        return []
    conn = _alignment_conn()
    query = (
        "SELECT timestamp, entry_id, violation_type, severity, patch_link, review_status FROM warnings WHERE 1=1"
    )
    params: List[Any] = []
    if min_severity is not None:
        query += " AND severity >= ?"
        params.append(min_severity)
    if max_severity is not None:
        query += " AND severity <= ?"
        params.append(max_severity)
    if review_status is not None:
        query += " AND review_status = ?"
        params.append(review_status)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [
        {
            "timestamp": row[0],
            "entry_id": row[1],
            "violation_type": row[2],
            "severity": row[3],
            "patch_link": row[4],
            "review_status": row[5],
        }
        for row in rows
    ]


def update_warning_status(entry_id: str, status: str) -> None:
    """Update the review status for a warning identified by ``entry_id``."""

    _ensure_log_dir()
    conn = _alignment_conn()
    conn.execute(
        "UPDATE warnings SET review_status = ? WHERE entry_id = ?",
        (status, entry_id),
    )
    conn.commit()


def mark_warning_pending(entry_id: str) -> None:
    """Mark a warning as pending review."""

    update_warning_status(entry_id, "pending")


def mark_warning_approved(entry_id: str) -> None:
    """Mark a warning as approved."""

    update_warning_status(entry_id, "approved")


def mark_warning_rejected(entry_id: str) -> None:
    """Mark a warning as rejected."""

    update_warning_status(entry_id, "rejected")


def log_violation(
    entry_id: str,
    violation_type: str,
    severity: int,
    evidence: Dict[str, Any],
    *,
    alignment_warning: bool = False,
) -> None:
    """Record a violation or alignment warning in the append-only log.

    Parameters
    ----------
    entry_id: str
        Unique identifier of the action or event that triggered the violation.
    violation_type: str
        Category of violation, e.g. ``"policy_violation"`` or ``"security_risk"``.
    severity: int
        Integer severity score. Higher values indicate more serious violations.
    evidence: dict
        Contextual information showing why the event was flagged. Should be
        JSON serialisable.
    alignment_warning: bool, optional
        When ``True`` the entry represents an alignment warning rather than a
        hard violation.  Warnings are also published on the event bus.
    """
    _ensure_log_dir()
    ev = dict(evidence)
    ev.setdefault("severity", severity)
    record = {
        "timestamp": int(time.time()),
        "entry_id": entry_id,
        "violation_type": violation_type,
        "severity": severity,
        "evidence": ev,
        "alignment_warning": alignment_warning,
    }
    file_logger = _get_logger(LOG_PATH)
    file_logger.info(json.dumps(record))

    if alignment_warning:
        try:
            persist_alignment_warning(record)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed persisting alignment warning: %s", exc)

    if alignment_warning and _event_bus is not None:
        payload = {
            "timestamp": record["timestamp"],
            "entry_id": entry_id,
            "violation_type": violation_type,
            "severity": severity,
            "evidence": evidence,
        }

        def _publish() -> None:
            try:
                publish_with_retry(
                    _event_bus, "alignment:flag", payload, delay=0.1
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("failed publishing alignment warning: %s", exc)

        Thread(target=_publish, daemon=True).start()


def load_recent_violations(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent *limit* violation records."""
    if limit <= 0:
        return []
    log_path = Path(LOG_PATH)
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    records: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_recent_alignment_warnings(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent *limit* alignment warnings."""
    if limit <= 0:
        return []
    log_path = Path(LOG_PATH)
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    warnings: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("alignment_warning"):
            warnings.append(rec)
            if len(warnings) >= limit:
                break
    warnings.reverse()
    return warnings


def recent_alignment_warnings(limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience wrapper returning recent alignment warnings.

    This helper provides a shorter public name while delegating to
    :func:`load_recent_alignment_warnings` for the actual retrieval logic.
    """

    return load_recent_alignment_warnings(limit)


def violation_summary(entry_id: str) -> str:
    """Return a brief summary of all violations for *entry_id*."""
    log_path = Path(LOG_PATH)
    if not log_path.exists():
        return "No violations logged."
    summaries = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("entry_id") == entry_id:
                ts = datetime.utcfromtimestamp(rec.get("timestamp", 0)).isoformat()
                vt = rec.get("violation_type")
                sev = rec.get("severity")
                summaries.append(f"{ts} - {vt} (severity {sev})")
    if not summaries:
        return f"No violations found for {entry_id}."
    joined = "\n".join(summaries)
    return f"Violation report for {entry_id}:\n{joined}"


__all__ = [
    "log_violation",
    "load_recent_violations",
    "load_recent_alignment_warnings",
    "persist_alignment_warning",
    "load_persisted_alignment_warnings",
    "update_warning_status",
    "mark_warning_pending",
    "mark_warning_approved",
    "mark_warning_rejected",
    "recent_alignment_warnings",
    "violation_summary",
    "set_event_bus",
]

