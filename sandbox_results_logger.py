"""Persist sandbox run metrics to the legacy SQLite backend.

The new :mod:`sandbox_runner.scoring` module aggregates metrics and stores
them in JSONL/summary form. To maintain backwards compatibility with the
older SQLite-based logger this module exposes :func:`record_run` which stores
an identical record in ``sandbox_data/run_metrics.db``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional during tests
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback for test envs
    from dynamic_path_router import resolve_path  # type: ignore


_LOG_DIR = Path(resolve_path("sandbox_data"))
_DB_PATH = _LOG_DIR / "run_metrics.db"
_lock = threading.Lock()


def record_run(metrics: Dict[str, Any]) -> None:
    """Persist *metrics* to the SQLite ``run_metrics`` table."""

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        try:
            with closing(conn.execute("PRAGMA journal_mode=WAL;")) as cursor:
                cursor.fetchall()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    ts TEXT,
                    success INTEGER,
                    runtime REAL,
                    entropy_delta REAL,
                    roi REAL,
                    coverage TEXT,
                    functions_hit INTEGER,
                    executed_functions TEXT,
                    error TEXT
                )
                """
            )
            serialised = {
                **metrics,
                "coverage": json.dumps(metrics.get("coverage"))
                if metrics.get("coverage") is not None
                else None,
                "executed_functions": json.dumps(metrics.get("executed_functions"))
                if metrics.get("executed_functions") is not None
                else None,
            }
            conn.execute(
                """
                INSERT INTO runs (
                    ts, success, runtime, entropy_delta, roi,
                    coverage, functions_hit, executed_functions, error
                ) VALUES (
                    :ts, :success, :runtime, :entropy_delta, :roi,
                    :coverage, :functions_hit, :executed_functions, :error
                )
                """,
                serialised,
            )
            conn.commit()
        finally:  # pragma: no cover - ensure connection closes
            conn.close()


def record_self_debug_metrics(metrics: Dict[str, Any]) -> None:
    """Persist self-debug metrics to the SQLite backend."""

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        try:
            with closing(conn.execute("PRAGMA journal_mode=WAL;")) as cursor:
                cursor.fetchall()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS self_debug_runs (
                    ts TEXT,
                    correlation_id TEXT,
                    source TEXT,
                    attempts INTEGER,
                    classification TEXT,
                    patch_validity TEXT,
                    roi_delta_total REAL,
                    roi_delta TEXT,
                    exit_reason TEXT
                )
                """
            )
            serialised = {
                **metrics,
                "classification": json.dumps(metrics.get("classification"))
                if metrics.get("classification") is not None
                else None,
                "patch_validity": json.dumps(metrics.get("patch_validity"))
                if metrics.get("patch_validity") is not None
                else None,
                "roi_delta": json.dumps(metrics.get("roi_delta"))
                if metrics.get("roi_delta") is not None
                else None,
            }
            conn.execute(
                """
                INSERT INTO self_debug_runs (
                    ts, correlation_id, source, attempts, classification,
                    patch_validity, roi_delta_total, roi_delta, exit_reason
                ) VALUES (
                    :ts, :correlation_id, :source, :attempts, :classification,
                    :patch_validity, :roi_delta_total, :roi_delta, :exit_reason
                )
                """,
                serialised,
            )
            conn.commit()
        finally:  # pragma: no cover - ensure connection closes
            conn.close()


__all__ = ["record_run", "record_self_debug_metrics"]
