"""Centralized, tamper-resistant audit logging for Security AI."""

from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

from db_router import GLOBAL_ROUTER, init_db_router

# Directory and default log paths
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
JSONL_PATH = os.path.join(LOG_DIR, "audit_log.jsonl")
SQLITE_PATH = os.path.join(LOG_DIR, "audit_log.db")


def _ensure_log_dir() -> None:
    """Create the log directory if missing."""
    os.makedirs(LOG_DIR, exist_ok=True)


def generate_event_id(event_type: str) -> str:
    """Return a unique event identifier."""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"{event_type}-{ts}-{suffix}"


def log_event(event_type: str, data: Dict[str, Any], jsonl_path: str = JSONL_PATH) -> str:
    """Append an event record to the JSONL audit log."""
    _ensure_log_dir()
    event_id = generate_event_id(event_type)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "event_id": event_id,
        "data": data,
    }
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return event_id


def export_to_csv(jsonl_path: str, csv_path: str) -> None:
    """Convert a JSONL audit log to a flat CSV file."""
    if not os.path.exists(jsonl_path):
        return
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
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
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _ensure_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not already exist."""
    cur = conn.cursor()
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


def log_to_sqlite(event_type: str, data: Dict[str, Any], db_path: str = SQLITE_PATH) -> str:
    """Store an event in the SQLite log."""
    _ensure_log_dir()
    event_id = generate_event_id(event_type)
    ts = datetime.utcnow().isoformat()
    router = GLOBAL_ROUTER or init_db_router("default")
    conn = router.get_connection("events")
    _ensure_db(conn)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (event_id, timestamp, event_type) VALUES (?, ?, ?)",
        (event_id, ts, event_type),
    )
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        cur.execute(
            "INSERT INTO event_data (event_id, key, value) VALUES (?, ?, ?)",
            (event_id, key, str(value)),
        )
    conn.commit()
    return event_id


def get_recent_events(limit: int = 100, jsonl_path: str = JSONL_PATH, db_path: str = SQLITE_PATH) -> List[Dict[str, Any]]:
    """Return the most recent events from the JSONL or SQLite log."""
    if limit <= 0:
        return []
    router = GLOBAL_ROUTER or init_db_router("default")
    conn = router.get_connection("events")
    _ensure_db(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT event_id, timestamp, event_type FROM events ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    events: List[Dict[str, Any]] = []
    for event_id, ts, etype in cur.fetchall():
        cur.execute("SELECT key, value FROM event_data WHERE event_id=?", (event_id,))
        data = {k: v for k, v in cur.fetchall()}
        try:
            for k, v in data.items():
                if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                    data[k] = json.loads(v)
        except Exception:
            pass
        events.append({"timestamp": ts, "event_type": etype, "event_id": event_id, "data": data})
    if events:
        return list(reversed(events))
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]  # type: ignore[arg-type]
    lines.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return lines[:limit][::-1]


__all__ = [
    "generate_event_id",
    "log_event",
    "export_to_csv",
    "log_to_sqlite",
    "get_recent_events",
]
