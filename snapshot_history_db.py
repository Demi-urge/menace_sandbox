from __future__ import annotations

"""Persistent store for snapshot history and regression records."""

from pathlib import Path
from typing import Any, Dict
import json
import time

from sandbox_settings import SandboxSettings
try:  # pragma: no cover - prefer namespaced package when available
    from menace_sandbox.self_improvement import prompt_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from self_improvement import prompt_memory  # type: ignore
from db_router import DBRouter

try:  # pragma: no cover - optional dependency location
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    def resolve_path(p: str) -> str:  # type: ignore
        return p


def _db_path(settings: SandboxSettings | None = None) -> Path:
    settings = settings or SandboxSettings()
    return Path(resolve_path(settings.sandbox_data_dir)) / "snapshot_history.db"


def _get_conn(settings: SandboxSettings | None = None):
    """Return a connection to the snapshot history database.

    The required tables are created on demand.
    """

    path = _db_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    router = DBRouter("snapshot_history", str(path), str(path))
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
    """Store *snap* for ``cycle`` and return the inserted row id."""

    conn = _get_conn()
    cur = conn.execute(
        (
            "INSERT INTO snapshots(" "cycle, stage, ts, roi, sandbox_score, entropy, "
            "call_graph_complexity, token_diversity, prompt, diff)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)"
        ),
        (
            int(cycle),
            stage,
            float(getattr(snap, "timestamp", time.time())),
            float(getattr(snap, "roi", 0.0)),
            float(getattr(snap, "sandbox_score", 0.0)),
            float(getattr(snap, "entropy", 0.0)),
            float(getattr(snap, "call_graph_complexity", 0.0)),
            float(getattr(snap, "token_diversity", 0.0)),
            getattr(snap, "prompt", None) or "",
            getattr(snap, "diff", None) or "",
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
    """Store ``delta`` for ``cycle`` and return the inserted row id."""

    conn = _get_conn()
    cur = conn.execute(
        (
            "INSERT INTO deltas(" "cycle, before_id, after_id, ts, roi_delta, "
            "sandbox_score_delta, entropy_delta, call_graph_complexity_delta, "
            "token_diversity_delta, regression) VALUES(" "?,?,?,?,?,?,?,?,?,?)"
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

    conn = _get_conn(settings)
    row = conn.execute(
        "SELECT cycle FROM deltas WHERE regression=0 ORDER BY cycle DESC LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else None


def log_regression(prompt: str | None, diff: str | None, delta: Dict[str, Any]) -> None:
    """Persist a regression record to the ``regressions`` table."""

    conn = _get_conn()
    ts = time.time()
    diff_text = diff or ""
    try:
        if diff and Path(diff).is_file():
            diff_text = Path(diff).read_text(encoding="utf-8")
    except Exception:  # pragma: no cover - best effort
        diff_text = diff or ""
    conn.execute(
        (
            "INSERT INTO regressions(" "ts, prompt, diff, roi_delta, entropy_delta, "
            "delta_json) VALUES(?,?,?,?,?,?)"
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
        try:  # pragma: no cover - best effort
            prompt_memory.record_downgrade(str(prompt))
        except Exception:
            pass


__all__ = [
    "record_snapshot",
    "record_delta",
    "last_successful_cycle",
    "log_regression",
]
