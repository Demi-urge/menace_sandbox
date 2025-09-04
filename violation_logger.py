"""Append-only violation logging utilities for Security AI."""

from __future__ import annotations

import json
import os
import time
import logging
from datetime import datetime
from threading import Thread, Lock
from typing import Any, List, Dict, Optional

from dynamic_path_router import resolve_dir, resolve_path

from .retry_utils import publish_with_retry
from db_router import GLOBAL_ROUTER
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
LOG_DIR = resolve_dir("logs")
LOG_PATH = resolve_path("logs/violation_log.jsonl")
ALIGNMENT_DB_PATH = resolve_path("logs/alignment_warnings.db")

logger = logging.getLogger(__name__)
_logger_lock = Lock()
_loggers: Dict[str, logging.Logger] = {}

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
    os.makedirs(LOG_DIR, exist_ok=True)


def _get_logger(path: str) -> logging.Logger:
    with _logger_lock:
        lg = _loggers.get(path)
        if lg is None:
            settings = SandboxSettings()
            lg = logging.getLogger(f"violation_logger_{path}")
            lg.setLevel(logging.INFO)
            rotate_seconds = settings.log_rotation_seconds
            if rotate_seconds:
                handler = LockedTimedRotatingFileHandler(
                    path,
                    when="s",
                    interval=rotate_seconds,
                    backupCount=settings.log_rotation_backup_count,
                    encoding="utf-8",
                )
            else:
                handler = LockedRotatingFileHandler(
                    path,
                    maxBytes=settings.log_rotation_max_bytes,
                    backupCount=settings.log_rotation_backup_count,
                    encoding="utf-8",
                )
            handler.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(handler)
            lg.propagate = False
            _loggers[path] = lg
        return lg


def _alignment_conn() -> sqlite3.Connection:
    """Return a connection for alignment warnings."""
    if GLOBAL_ROUTER is None:
        raise RuntimeError("Database router is not initialised")
    return GLOBAL_ROUTER.get_connection("errors")


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
    if GLOBAL_ROUTER is None and not os.path.exists(ALIGNMENT_DB_PATH):
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
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
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
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
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
    if not os.path.exists(LOG_PATH):
        return "No violations logged."
    summaries = []
    with open(LOG_PATH, "r", encoding="utf-8") as fh:
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

