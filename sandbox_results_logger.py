from __future__ import annotations

"""Lightweight logger for sandbox test results.

The :func:`record_run` function persists metrics about each sandbox execution to
both a JSONL file and a SQLite database.  This module is intentionally simple so
that it can be imported from lightweight test environments.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - allow flat import
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore

LOG_DIR = Path(resolve_path("sandbox_data"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = LOG_DIR / "sandbox_runs.jsonl"
DB_PATH = LOG_DIR / "sandbox_runs.db"

_lock = threading.Lock()
_db_initialised = False

def _init_db() -> None:
    """Initialise the SQLite store if necessary."""
    global _db_initialised
    if _db_initialised:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                ts TEXT,
                success INTEGER,
                entropy_delta REAL,
                runtime REAL,
                error TEXT,
                coverage TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary (
                key TEXT PRIMARY KEY,
                value REAL
            )
            """
        )
        conn.commit()
    _db_initialised = True


def record_run(metrics: Dict[str, Any]) -> None:
    """Persist a single sandbox execution *metrics*.

    Parameters
    ----------
    metrics:
        Dictionary containing run information.  Supported keys include
        ``success`` (bool), ``entropy_delta`` (float), ``runtime`` (float),
        ``error`` (str) and ``coverage`` (arbitrary JSON‑serialisable object).
    """
    _init_db()
    data = dict(metrics)
    data.setdefault("success", False)
    data.setdefault("entropy_delta", 0.0)
    data.setdefault("runtime", 0.0)
    ts = datetime.utcnow().isoformat()
    data["ts"] = ts
    cov = data.get("coverage")
    if cov is not None and not isinstance(cov, str):
        try:
            data["coverage"] = json.dumps(cov)
        except Exception:
            data["coverage"] = str(cov)
    try:
        success = 1 if data.get("success") else 0
        failure = 0 if success else 1
        with _lock:
            with open(JSONL_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(data) + "\n")
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO runs(ts, success, entropy_delta, runtime, error, coverage) VALUES(?,?,?,?,?,?)",
                    (
                        ts,
                        success,
                        float(data.get("entropy_delta", 0.0)),
                        float(data.get("runtime", 0.0)),
                        data.get("error"),
                        data.get("coverage"),
                    ),
                )
                # Update cumulative counts
                conn.execute(
                    "INSERT INTO summary(key, value) VALUES('successes', ?) ON CONFLICT(key) DO UPDATE SET value=value+?",
                    (success, success),
                )
                conn.execute(
                    "INSERT INTO summary(key, value) VALUES('failures', ?) ON CONFLICT(key) DO UPDATE SET value=value+?",
                    (failure, failure),
                )
                conn.execute(
                    "INSERT INTO summary(key, value) VALUES('runtime_total', ?) ON CONFLICT(key) DO UPDATE SET value=value+?",
                    (float(data.get("runtime", 0.0)), float(data.get("runtime", 0.0))),
                )
                conn.execute(
                    "INSERT INTO summary(key, value) VALUES('entropy_total', ?) ON CONFLICT(key) DO UPDATE SET value=value+?",
                    (float(data.get("entropy_delta", 0.0)), float(data.get("entropy_delta", 0.0))),
                )
                conn.commit()
    except Exception:
        # This logger is best‑effort; failures should not break the caller.
        pass


__all__ = ["record_run"]
