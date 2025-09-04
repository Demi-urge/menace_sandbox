from __future__ import annotations

"""Capture and compare self-improvement state metrics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .baseline_tracker import BaselineTracker
from ..sandbox_settings import SandboxSettings
from .sandbox_score import get_latest_sandbox_score
from . import metrics
from ..module_graph_analyzer import build_import_graph


@dataclass
class StateSnapshot:
    """Snapshot of repository metrics at a point in time."""

    roi: float
    sandbox_score: float
    entropy: float
    call_graph_edge_count: int
    token_diversity: float


# ---------------------------------------------------------------------------

def capture_state(repo_path: Path, tracker: BaselineTracker) -> StateSnapshot:
    """Collect current repository metrics into a :class:`StateSnapshot`."""

    settings = SandboxSettings()
    roi = float(tracker.current("roi"))
    sandbox_score = float(get_latest_sandbox_score(settings.sandbox_score_db))

    files = list(repo_path.rglob("*.py"))
    entropy = float(metrics.compute_code_entropy(files, settings=settings))

    try:
        _, _, _, _, _, diversity = metrics._collect_metrics(
            files, repo_path, settings
        )
    except Exception:  # pragma: no cover - best effort
        diversity = 0.0

    try:
        graph = build_import_graph(repo_path)
        edge_count = int(graph.number_of_edges())
    except Exception:  # pragma: no cover - best effort
        edge_count = 0

    return StateSnapshot(
        roi=roi,
        sandbox_score=sandbox_score,
        entropy=entropy,
        call_graph_edge_count=edge_count,
        token_diversity=float(diversity),
    )


# ---------------------------------------------------------------------------

def compare_snapshots(
    before: StateSnapshot, after: StateSnapshot
) -> Dict[str, float]:
    """Return per-metric differences between two snapshots ``after - before``."""

    return {
        "roi": after.roi - before.roi,
        "sandbox_score": after.sandbox_score - before.sandbox_score,
        "entropy": after.entropy - before.entropy,
        "call_graph_edge_count": after.call_graph_edge_count
        - before.call_graph_edge_count,
        "token_diversity": after.token_diversity - before.token_diversity,
    }


# ---------------------------------------------------------------------------
# Backwards compatibility aliases

Snapshot = StateSnapshot

def capture_snapshot(tracker: BaselineTracker, settings: SandboxSettings) -> StateSnapshot:
    """Compatibility wrapper forwarding to :func:`capture_state`."""

    return capture_state(Path(settings.sandbox_repo_path), tracker)


def delta(a: StateSnapshot, b: StateSnapshot) -> Dict[str, float]:
    """Compatibility wrapper for :func:`compare_snapshots`."""

    return compare_snapshots(a, b)


__all__ = [
    "StateSnapshot",
    "capture_state",
    "compare_snapshots",
    "Snapshot",
    "capture_snapshot",
    "delta",
]
