from __future__ import annotations

"""Persistent store for snapshot regression records."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict

from sandbox_settings import SandboxSettings

try:  # pragma: no cover - optional dependency location
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    def resolve_path(p: str) -> str:  # type: ignore
        return p


def _db_path(settings: SandboxSettings | None = None) -> Path:
    settings = settings or SandboxSettings()
    return Path(resolve_path(settings.sandbox_data_dir)) / "snapshot_history.db"


def log_regression(prompt: str | None, diff: str | None, delta: Dict[str, Any]) -> None:
    """Persist a regression record to :mod:`snapshot_history.db`."""

    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
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
        ts = time.time()
        diff_text = diff or ""
        try:
            if diff and Path(diff).is_file():
                diff_text = Path(diff).read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - best effort
            diff_text = diff or ""
        conn.execute(
            "INSERT INTO regressions(ts, prompt, diff, roi_delta, entropy_delta, delta_json) VALUES(?,?,?,?,?,?)",
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
    finally:
        conn.close()


__all__ = ["log_regression"]
