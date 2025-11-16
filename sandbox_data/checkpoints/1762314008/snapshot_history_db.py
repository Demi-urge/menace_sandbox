"""Persistent storage helpers for self-improvement snapshot data.

The real Menace stack records rich telemetry about each autonomous run. The
sandbox environment only requires a small subset of that functionality but it
must behave deterministically across platforms, including Windows. This module
provides a minimal SQLite-backed implementation that mirrors the production
interface so :mod:`run_autonomous` can execute end-to-end without optional cloud
services.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Tuple

try:  # pragma: no cover - prefer package-relative import
    from menace_sandbox.sandbox_settings import SandboxSettings
except ImportError:  # pragma: no cover - flat execution fallback
    from sandbox_settings import SandboxSettings  # type: ignore

try:  # pragma: no cover - prefer package-relative import
    from menace_sandbox.self_improvement import prompt_memory
except ImportError:  # pragma: no cover - flat execution fallback
    from self_improvement import prompt_memory  # type: ignore

try:  # pragma: no cover - prefer package-relative import
    from menace_sandbox.dynamic_path_router import resolve_path
except ImportError:  # pragma: no cover - flat execution fallback
    from dynamic_path_router import resolve_path  # type: ignore

from db_router import DBRouter

__all__ = [
    "record_snapshot",
    "record_delta",
    "log_regression",
    "last_successful_cycle",
]


_LOCK = RLock()
_ROUTER_CACHE: dict[Path, DBRouter] = {}


def _ensure_router(settings: SandboxSettings | None = None) -> Tuple[DBRouter, Path]:
    """Return a configured :class:`DBRouter` and the database path."""

    settings = settings or SandboxSettings()
    sandbox_dir = Path(resolve_path(getattr(settings, "sandbox_data_dir", ".")))
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    db_path = sandbox_dir / "snapshot_history.db"

    with _LOCK:
        router = _ROUTER_CACHE.get(db_path)
        if router is None:
            router = DBRouter("snapshot_history", str(db_path), str(db_path))
            _ROUTER_CACHE[db_path] = router
    return router, db_path


def _get_connection(settings: SandboxSettings | None = None):
    router, _ = _ensure_router(settings)
    conn = router.get_connection("history")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle INTEGER,
            stage TEXT,
            ts REAL,
            roi REAL,
            sandbox_score REAL,
            entropy REAL,
            call_graph_complexity REAL,
            token_diversity REAL,
            prompt TEXT,
            diff TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deltas(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle INTEGER,
            before_id INTEGER,
            after_id INTEGER,
            ts REAL,
            roi_delta REAL,
            sandbox_score_delta REAL,
            entropy_delta REAL,
            call_graph_complexity_delta REAL,
            token_diversity_delta REAL,
            regression INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS regressions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            prompt TEXT,
            diff TEXT,
            roi_delta REAL,
            entropy_delta REAL,
            delta_json TEXT
        )
        """
    )
    return conn


def record_snapshot(cycle: int, stage: str, snap: Any) -> int:
    """Persist ``snap`` for ``cycle`` and return the inserted row id."""

    conn = _get_connection()
    payload = {
        "roi": float(getattr(snap, "roi", 0.0)),
        "sandbox_score": float(getattr(snap, "sandbox_score", 0.0)),
        "entropy": float(getattr(snap, "entropy", 0.0)),
        "call_graph_complexity": float(getattr(snap, "call_graph_complexity", 0.0)),
        "token_diversity": float(getattr(snap, "token_diversity", 0.0)),
    }
    prompt = getattr(snap, "prompt", "")
    if isinstance(prompt, dict):
        prompt = json.dumps(prompt)
    diff = getattr(snap, "diff", "")
    if diff is None:
        diff = ""

    cur = conn.execute(
        (
            "INSERT INTO snapshots(" "cycle, stage, ts, roi, sandbox_score, entropy, "
            "call_graph_complexity, token_diversity, prompt, diff) VALUES(?,?,?,?,?,?,?,?,?,?)"
        ),
        (
            int(cycle),
            stage,
            float(getattr(snap, "timestamp", time.time())),
            payload["roi"],
            payload["sandbox_score"],
            payload["entropy"],
            payload["call_graph_complexity"],
            payload["token_diversity"],
            prompt or "",
            diff,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def record_delta(
    cycle: int,
    before_id: int,
    after_id: int,
    delta: Dict[str, Any],
    ts: float | None = None,
) -> int:
    """Persist ``delta`` for ``cycle`` and return the inserted row id."""

    conn = _get_connection()
    cur = conn.execute(
        (
            "INSERT INTO deltas(" "cycle, before_id, after_id, ts, roi_delta, "
            "sandbox_score_delta, entropy_delta, call_graph_complexity_delta, "
            "token_diversity_delta, regression) VALUES(?,?,?,?,?,?,?,?,?,?)"
        ),
        (
            int(cycle),
            int(before_id),
            int(after_id),
            float(ts if ts is not None else time.time()),
            float(delta.get("roi", 0.0)),
            float(delta.get("sandbox_score", 0.0)),
            float(delta.get("entropy", 0.0)),
            float(delta.get("call_graph_complexity", 0.0)),
            float(delta.get("token_diversity", 0.0)),
            1 if bool(delta.get("regression")) else 0,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def last_successful_cycle(settings: SandboxSettings | None = None) -> int | None:
    """Return the most recent cycle id without a regression."""

    conn = _get_connection(settings)
    row = conn.execute(
        "SELECT cycle FROM deltas WHERE regression=0 ORDER BY cycle DESC LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else None


def log_regression(prompt: str | None, diff: str | None, delta: Dict[str, Any]) -> None:
    """Persist a regression record to the ``regressions`` table."""

    conn = _get_connection()
    ts = time.time()
    diff_text = diff or ""
    try:
        diff_path = Path(diff_text)
        if diff_path.exists() and diff_path.is_file():
            diff_text = diff_path.read_text(encoding="utf-8")
    except Exception:  # pragma: no cover - best effort for unusual paths
        pass

    conn.execute(
        (
            "INSERT INTO regressions(" "ts, prompt, diff, roi_delta, entropy_delta, delta_json)"
            " VALUES(?,?,?,?,?,?)"
        ),
        (
            ts,
            prompt or "",
            diff_text,
            float(delta.get("roi", 0.0)),
            float(delta.get("entropy", 0.0)),
            json.dumps(delta),
        ),
    )
    conn.commit()

    if prompt:
        try:  # pragma: no cover - optional downgrade logging
            prompt_memory.record_downgrade(str(prompt))
        except Exception:
            pass
