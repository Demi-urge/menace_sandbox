from __future__ import annotations

"""Capture and compare self-improvement state metrics."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import json
import logging
import shutil
import time

import codebase_diff_checker
import logging_utils
from . import prompt_memory
try:  # pragma: no cover - optional dependency location
    from ..dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore
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


class SnapshotTracker:
    """Track state snapshots across self-improvement iterations."""

    def __init__(self, repo_path: Path, tracker: BaselineTracker) -> None:
        self.repo_path = Path(repo_path)
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
        base = Path(resolve_path(SandboxSettings().sandbox_data_dir)) / "snapshots"
        self._persist_path = base / "last_snapshot.json"
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self.last_snapshot = StateSnapshot(**data)
        except Exception:
            self.last_snapshot = None

    # --------------------------------------------------------------
    def _persist_last_snapshot(self) -> None:
        if self.last_snapshot is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(asdict(self.last_snapshot)), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - best effort
            self.logger.debug("failed to persist snapshot", exc_info=True)

    def store(self, snap: StateSnapshot) -> None:
        self.last_snapshot = snap
        self._persist_last_snapshot()

    def capture(self) -> StateSnapshot:
        """Capture and store the current repository snapshot."""

        snap = capture_state(self.repo_path, self.tracker)
        self.store(snap)
        return snap

    def evaluate_change(
        self, after: StateSnapshot, prompt: Any, diff_path: str | Path
    ) -> Dict[str, float]:
        """Evaluate ``after`` against the previous snapshot.

        If ROI decreased or entropy increased a warning is logged, the prompt
        attempt recorded and a diff snapshot stored under
        ``sandbox_data/diffs/``.

        When all metric deltas are positive the changed modules referenced by
        ``diff_path`` are copied to ``sandbox_data/checkpoints/<ts>/`` and the
        confidence score for the associated strategy is incremented.
        """

        before = self.last_snapshot
        self.store(after)
        if before is None:
            return {}

        delta = compare_snapshots(before, after)

        if all(v > 0 for v in delta.values()):
            try:
                settings = SandboxSettings()
                base = Path(resolve_path(settings.sandbox_data_dir))
                ckpt_dir = base / "checkpoints" / str(int(time.time()))

                files: list[Path] = []
                try:
                    diff_text = Path(diff_path).read_text(encoding="utf-8")
                    for line in diff_text.splitlines():
                        if line.startswith("+++ b/"):
                            name = line[6:]
                            if name != "/dev/null":
                                files.append(Path(name))
                except Exception:
                    files = []

                for rel in files:
                    src = self.repo_path / rel
                    dest = ckpt_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src, dest)
                    except Exception:
                        try:
                            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                        except Exception:
                            pass

                strategy: str | None = None
                if isinstance(prompt, dict):
                    strategy = (
                        prompt.get("strategy") or prompt.get("strategy_name")
                    )
                elif hasattr(prompt, "strategy"):
                    strategy = getattr(prompt, "strategy")
                elif isinstance(prompt, str):
                    strategy = prompt

                if strategy:
                    conf_path = base / "strategy_confidence.json"
                    try:
                        conf = json.loads(conf_path.read_text(encoding="utf-8"))
                        if not isinstance(conf, dict):
                            conf = {}
                    except Exception:
                        conf = {}
                    conf[str(strategy)] = int(conf.get(str(strategy), 0)) + 1
                    conf_path.parent.mkdir(parents=True, exist_ok=True)
                    conf_path.write_text(json.dumps(conf), encoding="utf-8")
            except Exception:  # pragma: no cover - best effort
                pass

        elif delta.get("roi", 0) < 0 or delta.get("entropy", 0) > 0:
            try:  # pragma: no cover - best effort logging
                prompt_memory.log_prompt_attempt(
                    prompt, success=False, exec_result=None, roi_meta=delta
                )
            except Exception:
                pass

            try:  # pragma: no cover - diff storage best effort
                settings = SandboxSettings()
                diff_dir = Path(resolve_path(settings.sandbox_data_dir)) / "diffs"
                diff_dir.mkdir(parents=True, exist_ok=True)
                output = diff_dir / Path(diff_path)
                codebase_diff_checker.compare_snapshots(before, after, output)
            except Exception:
                pass

            self.logger.warning(
                "negative roi or increased entropy",
                extra=logging_utils.log_record(delta=delta),
            )

        return delta


# ---------------------------------------------------------------------------
# Backwards compatibility aliases

Snapshot = StateSnapshot


def capture_snapshot(tracker: BaselineTracker, settings: SandboxSettings) -> StateSnapshot:
    """Compatibility wrapper forwarding to :func:`capture_state`."""

    return capture_state(Path(resolve_path(settings.sandbox_repo_path)), tracker)


def delta(a: StateSnapshot, b: StateSnapshot) -> Dict[str, float]:
    """Compatibility wrapper for :func:`compare_snapshots`."""

    return compare_snapshots(a, b)


__all__ = [
    "StateSnapshot",
    "capture_state",
    "compare_snapshots",
    "SnapshotTracker",
    "Snapshot",
    "capture_snapshot",
    "delta",
]
