from __future__ import annotations

"""Capture and persist self‑improvement cycle snapshots."""

from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
from typing import Dict, Sequence

from .baseline_tracker import TRACKER as BASELINE_TRACKER
from .metrics import collect_snapshot_metrics, compute_call_graph_complexity
from ..sandbox_settings import SandboxSettings

try:  # pragma: no cover - optional dependency location
    from ..dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - optional module
    from .. import relevancy_radar
except Exception:  # pragma: no cover
    relevancy_radar = None  # type: ignore


@dataclass
class Snapshot:
    """Point in time metrics for a self‑improvement cycle."""

    roi: float
    sandbox_score: float
    entropy: float
    call_graph_complexity: float
    token_diversity: float
    prompt: str | None = None
    diff: str | None = None
    timestamp: float = 0.0


_cycle_id = 0


def _snapshot_path(settings: SandboxSettings, cycle_id: int, stage: str) -> Path:
    base = Path(resolve_path(settings.sandbox_data_dir)) / "snapshots"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{cycle_id}_{stage}.json"


def capture(
    stage: str,
    files: Sequence[Path | str],
    roi: float,
    sandbox_score: float,
    prompt: str | None = None,
    diff: str | None = None,
) -> Snapshot:
    """Capture a snapshot for the given cycle *stage* and persist it."""

    global _cycle_id
    if stage.lower() == "pre":
        _cycle_id += 1

    settings = SandboxSettings()

    try:
        entropy, token_diversity = collect_snapshot_metrics(files, settings=settings)
    except Exception:  # pragma: no cover - best effort
        entropy, token_diversity = 0.0, 0.0

    try:
        if relevancy_radar and hasattr(relevancy_radar, "call_graph_complexity"):
            call_complexity = float(relevancy_radar.call_graph_complexity(files))
        else:  # fallback
            repo = Path(settings.sandbox_repo_path)
            call_complexity = compute_call_graph_complexity(repo)
    except Exception:  # pragma: no cover - best effort
        call_complexity = 0.0

    try:
        BASELINE_TRACKER.update(
            roi=float(roi),
            sandbox_score=float(sandbox_score),
            entropy=float(entropy),
            call_graph_complexity=call_complexity,
            token_diversity=token_diversity,
            record_momentum=stage.lower() == "post",
        )
    except Exception:  # pragma: no cover - best effort
        pass

    snap = Snapshot(
        roi=float(roi),
        sandbox_score=float(sandbox_score),
        entropy=float(entropy),
        call_graph_complexity=float(call_complexity),
        token_diversity=float(token_diversity),
        prompt=prompt,
        diff=diff,
        timestamp=time.time(),
    )

    path = _snapshot_path(settings, _cycle_id, stage)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(snap), fh)

    return snap


def compute_delta(prev: Snapshot, curr: Snapshot) -> Dict[str, float]:
    """Return per‑metric differences between two snapshots ``curr - prev``."""

    return {
        "roi": curr.roi - prev.roi,
        "sandbox_score": curr.sandbox_score - prev.sandbox_score,
        "entropy": curr.entropy - prev.entropy,
        "call_graph_complexity": curr.call_graph_complexity - prev.call_graph_complexity,
        "token_diversity": curr.token_diversity - prev.token_diversity,
        "timestamp": curr.timestamp - prev.timestamp,
    }


__all__ = ["Snapshot", "capture", "compute_delta"]
