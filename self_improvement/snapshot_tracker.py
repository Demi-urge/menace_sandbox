from __future__ import annotations

"""Capture and persist self‑improvement cycle snapshots."""

from dataclasses import dataclass, asdict
import json
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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

# ---------------------------------------------------------------------------
# Persistent downgrade tracking

settings = SandboxSettings()
_downgrade_path = Path(resolve_path(settings.sandbox_data_dir)) / "prompt_downgrades.json"
try:
    _downgrade_counts_raw = json.loads(_downgrade_path.read_text(encoding="utf-8"))
    downgrade_counts: Dict[str, int] = {
        str(k): int(v) for k, v in _downgrade_counts_raw.items()
        if isinstance(k, str)
    }
except Exception:
    downgrade_counts = {}


def _save_downgrades() -> None:
    """Persist :data:`downgrade_counts` to disk."""

    try:
        _downgrade_path.parent.mkdir(parents=True, exist_ok=True)
        _downgrade_path.write_text(json.dumps(downgrade_counts), encoding="utf-8")
    except Exception:  # pragma: no cover - best effort
        pass


def record_downgrade(name: str) -> int:
    """Increment downgrade counter for ``name`` and persist it."""

    count = downgrade_counts.get(name, 0) + 1
    downgrade_counts[name] = count
    _save_downgrades()
    return count


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


# ---------------------------------------------------------------------------


class SnapshotTracker:
    """Maintain before/after snapshots and expose metric deltas."""

    def __init__(self) -> None:
        self._snaps: dict[str, Snapshot] = {}
        self._context: dict[str, Mapping[str, Any]] = {}

    def capture(self, stage: str, context: Mapping[str, Any]) -> Snapshot:
        files = context.get("files", [])
        roi = float(context.get("roi", 0.0))
        score = float(context.get("sandbox_score", 0.0))
        prompt = context.get("prompt")
        diff = context.get("diff")
        snap = capture(
            stage=stage,
            files=files,
            roi=roi,
            sandbox_score=score,
            prompt=prompt if isinstance(prompt, str) else None,
            diff=diff if isinstance(diff, str) else None,
        )
        self._snaps[stage] = snap
        self._context[stage] = dict(context)
        return snap

    def delta(self) -> Dict[str, float]:
        before = self._snaps.get("before") or self._snaps.get("pre")
        after = self._snaps.get("after") or self._snaps.get("post")
        if before and after:
            return compute_delta(before, after)
        return {}

def save_checkpoint(module_path: Path | str, cycle_id: str) -> Path:
    """Copy *module_path* to a checkpoint named after ``cycle_id``.

    The file is stored under ``sandbox_data/checkpoints/<module>/<cycle_id>.py``.
    Returns the destination path.
    """

    module_path = Path(module_path)
    base = Path(resolve_path(settings.sandbox_data_dir)) / "checkpoints" / module_path.stem
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"{cycle_id}{module_path.suffix}"
    try:
        shutil.copy2(module_path, dest)
    except Exception:  # pragma: no cover - best effort
        dest.write_text(module_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


__all__ = [
    "Snapshot",
    "SnapshotTracker",
    "capture",
    "compute_delta",
    "save_checkpoint",
    "downgrade_counts",
    "record_downgrade",
]
