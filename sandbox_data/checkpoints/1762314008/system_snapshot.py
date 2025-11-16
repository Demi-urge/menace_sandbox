from __future__ import annotations

"""System level snapshot utilities.

This module captures high level system metrics such as ROI, sandbox score and
code statistics so that self-improvement cycles can be evaluated over time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import time

from menace_sandbox.sandbox_settings import SandboxSettings
from .baseline_tracker import BaselineTracker
from .sandbox_score import get_latest_sandbox_score
from . import metrics as _si_metrics
from .metrics import compute_call_graph_complexity

try:  # pragma: no cover - sandbox results logger is optional
    import sandbox_results_logger  # type: ignore
except Exception:  # pragma: no cover
    sandbox_results_logger = None  # type: ignore


@dataclass
class SystemSnapshot:
    """Snapshot of system metrics at a point in time."""

    roi: float
    sandbox_score: float
    entropy: float
    call_graph_complexity: float
    token_diversity: float
    timestamp: float
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------

def _get_sandbox_score(settings: SandboxSettings) -> float:
    """Return the latest sandbox score.

    Preference is given to the :mod:`sandbox_results_logger` summary when
    available, otherwise the configured SQLite database is consulted.
    """

    score = 0.0
    if sandbox_results_logger is not None:
        load_summary = getattr(sandbox_results_logger, "load_summary", None)
        if callable(load_summary):
            try:
                summary = load_summary()  # type: ignore[misc]
                score = float(
                    summary.get("sandbox_score")
                    or summary.get("score")
                    or 0.0
                )
            except Exception:  # pragma: no cover - best effort
                score = 0.0
    if score == 0.0:
        try:
            score = float(get_latest_sandbox_score(settings.sandbox_score_db))
        except Exception:  # pragma: no cover - best effort
            score = 0.0
    return score


def capture_snapshot(engine: Any) -> SystemSnapshot:
    """Collect current metrics from *engine* into a :class:`SystemSnapshot`."""

    settings = SandboxSettings()
    repo_path = Path(settings.sandbox_repo_path)

    tracker: BaselineTracker = getattr(engine, "roi_tracker")
    roi = float(tracker.current("roi"))
    entropy = float(tracker.current("entropy"))
    sandbox_score = _get_sandbox_score(settings)

    files = list(repo_path.rglob("*.py"))
    try:
        _, _, _, _, _, token_diversity = _si_metrics._collect_metrics(
            files, repo_path, settings
        )
    except Exception:  # pragma: no cover - best effort
        token_diversity = 0.0

    try:
        call_complexity = compute_call_graph_complexity(repo_path)
    except Exception:  # pragma: no cover - best effort
        call_complexity = 0.0

    metadata = {
        "prompt": getattr(engine, "last_prompt", None)
        or getattr(getattr(engine, "self_coding_engine", None), "_last_prompt", None),
        "module_paths": getattr(engine, "module_paths", None)
        or getattr(engine, "last_module_paths", None),
    }

    return SystemSnapshot(
        roi=roi,
        sandbox_score=sandbox_score,
        entropy=entropy,
        call_graph_complexity=call_complexity,
        token_diversity=float(token_diversity),
        timestamp=float(time.time()),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------

def compare_snapshots(before: SystemSnapshot, after: SystemSnapshot) -> Dict[str, float]:
    """Return per-metric differences between two snapshots ``after - before``."""

    return {
        "roi": after.roi - before.roi,
        "sandbox_score": after.sandbox_score - before.sandbox_score,
        "entropy": after.entropy - before.entropy,
        "call_graph_complexity": after.call_graph_complexity
        - before.call_graph_complexity,
        "token_diversity": after.token_diversity - before.token_diversity,
    }


__all__ = ["SystemSnapshot", "capture_snapshot", "compare_snapshots"]
