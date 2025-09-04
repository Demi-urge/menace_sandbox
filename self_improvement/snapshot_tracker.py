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
from ..audit_logger import log_event as audit_log_event
try:  # pragma: no cover - optional dependency location
    from ..snapshot_history_db import (
        log_regression,
        record_snapshot,
        record_delta,
    )
except Exception:  # pragma: no cover
    from snapshot_history_db import (  # type: ignore
        log_regression,
        record_snapshot,
        record_delta,
    )

try:  # pragma: no cover - optional dependency location
    from ..dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - optional module
    from .. import relevancy_radar
except Exception:  # pragma: no cover
    relevancy_radar = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..module_index_db import ModuleIndexDB
except Exception:  # pragma: no cover
    ModuleIndexDB = None  # type: ignore


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
    id: int | None = None


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

    metrics = set(getattr(settings, "snapshot_metrics", []))

    entropy = token_diversity = 0.0
    if {"entropy", "token_diversity"} & metrics:
        try:
            entropy, token_diversity = collect_snapshot_metrics(files, settings=settings)
        except Exception:  # pragma: no cover - best effort
            entropy, token_diversity = 0.0, 0.0

    call_complexity = 0.0
    if "call_graph_complexity" in metrics:
        try:
            if relevancy_radar and hasattr(relevancy_radar, "call_graph_complexity"):
                call_complexity = float(relevancy_radar.call_graph_complexity(files))
            else:  # fallback
                repo = Path(settings.sandbox_repo_path)
                call_complexity = compute_call_graph_complexity(repo)
        except Exception:  # pragma: no cover - best effort
            call_complexity = 0.0

    to_update: dict[str, float] = {
        "roi": float(roi),
        "sandbox_score": float(sandbox_score),
    }
    if "entropy" in metrics:
        to_update["entropy"] = float(entropy)
    if "call_graph_complexity" in metrics:
        to_update["call_graph_complexity"] = float(call_complexity)
    if "token_diversity" in metrics:
        to_update["token_diversity"] = float(token_diversity)

    try:
        BASELINE_TRACKER.update(record_momentum=stage.lower() == "post", **to_update)
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
    try:
        snap.id = record_snapshot(_cycle_id, stage, snap)
    except Exception:  # pragma: no cover - best effort
        pass
    return snap


def compute_delta(prev: Snapshot, curr: Snapshot) -> Dict[str, float]:
    """Return per‑metric differences between two snapshots ``curr - prev``."""

    metrics = set(SandboxSettings().snapshot_metrics)
    delta: Dict[str, float] = {
        m: float(getattr(curr, m, 0.0)) - float(getattr(prev, m, 0.0)) for m in metrics
    }
    delta["timestamp"] = curr.timestamp - prev.timestamp
    return delta


# ---------------------------------------------------------------------------


class SnapshotTracker:
    """Maintain before/after snapshots and expose metric deltas."""

    def __init__(self) -> None:
        self._snaps: dict[str, Snapshot] = {}
        self._context: dict[str, Mapping[str, Any]] = {}
        settings = SandboxSettings()
        base = Path(resolve_path(settings.sandbox_data_dir))
        self._conf_path = base / "strategy_confidence.json"
        try:
            data = json.loads(self._conf_path.read_text(encoding="utf-8"))
            self.strategy_confidence: Dict[str, int] = {
                str(k): int(v) for k, v in data.items() if isinstance(k, str)
            }
        except Exception:
            self.strategy_confidence = {}
        self._module_map_path = base / "module_checkpoints.json"
        try:
            mapping = json.loads(self._module_map_path.read_text(encoding="utf-8"))
            self._module_map: Dict[str, Dict[str, str]] = {
                str(k): {str(sk): str(sv) for sk, sv in v.items()}
                for k, v in mapping.items()
                if isinstance(v, dict)
            }
        except Exception:
            self._module_map = {}

    def _save_conf(self) -> None:
        try:
            self._conf_path.parent.mkdir(parents=True, exist_ok=True)
            self._conf_path.write_text(
                json.dumps(self.strategy_confidence), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - best effort
            pass

    def _save_module_map(self) -> None:
        try:
            self._module_map_path.parent.mkdir(parents=True, exist_ok=True)
            self._module_map_path.write_text(
                json.dumps(self._module_map), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - best effort
            pass

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
        if not (before and after):
            return {}

        delta = compute_delta(before, after)
        roi_delta = float(delta.get("roi", 0.0))
        entropy_delta = float(delta.get("entropy", 0.0))
        regression = roi_delta < 0 or entropy_delta > 0
        delta["regression"] = regression
        if before.id is not None and after.id is not None:
            try:
                record_delta(_cycle_id, before.id, after.id, delta, after.timestamp)
            except Exception:  # pragma: no cover - best effort
                pass
        ctx = self._context.get("after") or self._context.get("post") or {}
        if regression:
            prompt = ctx.get("prompt")
            diff = ctx.get("diff") or ctx.get("diff_path")
            try:
                log_regression(prompt if isinstance(prompt, str) else None, diff, delta)
            except Exception:  # pragma: no cover - best effort
                pass
            try:
                audit_log_event(
                    "snapshot_regression",
                    {
                        "prompt": prompt,
                        "diff": diff,
                        "delta": {k: v for k, v in delta.items() if k != "regression"},
                    },
                )
            except Exception:  # pragma: no cover - best effort
                pass
        else:
            metrics = set(SandboxSettings().snapshot_metrics)
            positive = metrics - {"entropy"}
            improved = all(delta.get(m, 0.0) >= 0.0 for m in positive)
            if "entropy" in metrics:
                improved = improved and delta.get("entropy", 0.0) <= 0.0
            if improved:
                files = ctx.get("files") or []
                prompt = ctx.get("prompt")
                strategy: str | None = None
                if isinstance(prompt, dict):
                    strategy = prompt.get("strategy") or prompt.get("strategy_name")
                elif hasattr(prompt, "strategy"):
                    strategy = getattr(prompt, "strategy")
                elif isinstance(prompt, str):
                    strategy = prompt
                timestamp = str(int(time.time()))
                base = Path(resolve_path(SandboxSettings().sandbox_data_dir))
                ckpt_dir = base / "checkpoints" / timestamp
                for f in files:
                    try:
                        src = Path(f)
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        dest = ckpt_dir / src.name
                        try:
                            shutil.copy2(src, dest)
                        except Exception:  # pragma: no cover - best effort
                            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                        if ModuleIndexDB:
                            try:
                                ModuleIndexDB().get(str(src))
                            except Exception:
                                pass
                        if strategy:
                            rel = Path("checkpoints") / timestamp / src.name
                            self._module_map.setdefault(src.name, {})[str(strategy)] = rel.as_posix()
                    except Exception:  # pragma: no cover - best effort
                        pass
                if strategy:
                    self.strategy_confidence[str(strategy)] = int(
                        self.strategy_confidence.get(str(strategy), 0)
                    ) + 1
                    self._save_conf()
                    self._save_module_map()
        return delta

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


def get_best_checkpoint(module: Path | str) -> Path | None:
    """Return path to highest confidence checkpoint for ``module`` if any."""

    settings = SandboxSettings()
    base = Path(resolve_path(settings.sandbox_data_dir))
    conf_path = base / "strategy_confidence.json"
    map_path = base / "module_checkpoints.json"
    try:
        conf = json.loads(conf_path.read_text(encoding="utf-8"))
        if not isinstance(conf, dict):
            conf = {}
    except Exception:
        conf = {}
    try:
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            mapping = {}
    except Exception:
        mapping = {}
    key = Path(module).name
    strategies = mapping.get(key, {})
    if not strategies:
        return None
    best = max(strategies, key=lambda s: conf.get(s, 0))
    rel = Path(strategies[best])
    return base / rel


__all__ = [
    "Snapshot",
    "SnapshotTracker",
    "capture",
    "compute_delta",
    "save_checkpoint",
    "get_best_checkpoint",
    "downgrade_counts",
    "record_downgrade",
]
