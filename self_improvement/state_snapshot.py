from __future__ import annotations

"""Helpers for capturing and persisting self‑improvement state snapshots."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict
import sqlite3
import time

from .baseline_tracker import BaselineTracker
from ..sandbox_settings import SandboxSettings

try:  # pragma: no cover - optional dependency location
    from ..dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore

from .metrics import compute_call_graph_complexity, compute_entropy_metrics


@dataclass
class Snapshot:
    """Point in time state for self‑improvement metrics."""

    roi: float
    sandbox_score: float
    entropy: float
    call_graph_complexity: float
    token_diversity: float
    timestamp: float


# ---------------------------------------------------------------------------

def _latest_sandbox_score(path: str | Path) -> float:
    """Return most recent sandbox score from *path*.

    Returns ``0`` when the database or expected tables are missing.
    """

    try:
        conn = sqlite3.connect(str(resolve_path(str(path))))
    except Exception:
        return 0.0
    try:
        for query in (
            "SELECT score FROM sandbox_scores ORDER BY timestamp DESC LIMIT 1",
            "SELECT score FROM score_history ORDER BY ts DESC LIMIT 1",
        ):
            try:
                cur = conn.execute(query)
                row = cur.fetchone()
                if row:
                    return float(row[0])
            except Exception:
                continue
    finally:
        conn.close()
    return 0.0


def _token_diversity(repo: Path, settings: SandboxSettings) -> float:
    """Return average token diversity for *repo* using metrics helper."""
    files = list(repo.rglob("*.py"))
    try:
        _, _, avg_div = compute_entropy_metrics(files, settings=settings)
    except Exception:  # pragma: no cover - best effort
        avg_div = 0.0
    return float(avg_div)


# ---------------------------------------------------------------------------

def capture_snapshot(tracker: BaselineTracker, settings: SandboxSettings) -> Snapshot:
    """Collect current sandbox metrics into a :class:`Snapshot`."""

    repo = Path(settings.sandbox_repo_path)
    roi = float(tracker.current("roi"))
    entropy = float(tracker.current("entropy"))
    sandbox_score = _latest_sandbox_score(settings.sandbox_score_db)

    try:
        call_complexity = compute_call_graph_complexity(repo)
    except Exception:  # pragma: no cover - fall back if analysis fails
        call_complexity = 0.0

    diversity = _token_diversity(repo, settings)

    try:
        tracker.update(
            call_graph_complexity=call_complexity,
            token_diversity=diversity,
            record_momentum=False,
        )
    except Exception:  # pragma: no cover - best effort
        pass

    return Snapshot(
        roi=roi,
        sandbox_score=sandbox_score,
        entropy=entropy,
        call_graph_complexity=call_complexity,
        token_diversity=diversity,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------

def delta(a: Snapshot, b: Snapshot) -> Dict[str, float]:
    """Return per‑metric differences between two snapshots ``b - a``."""

    return {
        "roi": b.roi - a.roi,
        "sandbox_score": b.sandbox_score - a.sandbox_score,
        "entropy": b.entropy - a.entropy,
        "call_graph_complexity": b.call_graph_complexity - a.call_graph_complexity,
        "token_diversity": b.token_diversity - a.token_diversity,
        "timestamp": b.timestamp - a.timestamp,
    }


# ---------------------------------------------------------------------------

def save_snapshot(snapshot: Snapshot, settings: SandboxSettings) -> Path:
    """Persist *snapshot* under ``sandbox_data/snapshots`` and return the file path."""

    base = Path(resolve_path(settings.sandbox_data_dir)) / "snapshots"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{int(snapshot.timestamp)}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(snapshot), fh)
    return path


def load_snapshot(settings: SandboxSettings, timestamp: float | None = None) -> Snapshot:
    """Load a previously saved :class:`Snapshot`.

    When ``timestamp`` is ``None`` the most recent snapshot is returned.
    """

    base = Path(resolve_path(settings.sandbox_data_dir)) / "snapshots"
    if timestamp is None:
        candidates = sorted(base.glob("*.json"))
        if not candidates:
            raise FileNotFoundError("no snapshots available")
        path = candidates[-1]
    else:
        path = base / f"{int(timestamp)}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return Snapshot(**data)


__all__ = [
    "Snapshot",
    "capture_snapshot",
    "delta",
    "save_snapshot",
    "load_snapshot",
]

