from types import ModuleType, SimpleNamespace
from pathlib import Path
import importlib
import sys

import pytest

# Set up minimal package structure and stubs before importing modules
pkg_path = Path(__file__).resolve().parent.parent
root_pkg = ModuleType("menace_sandbox")
root_pkg.__path__ = [str(pkg_path.parent)]
sys.modules.setdefault("menace_sandbox", root_pkg)
sub_pkg = ModuleType("menace_sandbox.self_improvement")
sub_pkg.__path__ = [str(pkg_path)]
sys.modules.setdefault("menace_sandbox.self_improvement", sub_pkg)
sys.modules.setdefault("self_improvement", sub_pkg)

sys.modules.setdefault(
    "dynamic_path_router", SimpleNamespace(resolve_path=lambda p: Path(p))
)

from menace_sandbox.self_improvement.baseline_tracker import BaselineTracker
state_snapshot = importlib.import_module(
    "menace_sandbox.self_improvement.state_snapshot"
)


def _patch_helpers(monkeypatch, entropies, diversities, edges, scores):
    ent_iter = iter(entropies)
    div_iter = iter(diversities)
    edge_iter = iter(edges)
    score_iter = iter(scores)

    monkeypatch.setattr(
        state_snapshot.metrics,
        "compute_code_entropy",
        lambda files, settings=None: next(ent_iter),
    )

    def fake_collect(files, repo, settings=None):
        return {}, 0, 0, 0, 0, next(div_iter)

    monkeypatch.setattr(state_snapshot.metrics, "_collect_metrics", fake_collect)

    class Graph:
        def __init__(self, count):
            self._count = count

        def number_of_edges(self):
            return self._count

    monkeypatch.setattr(
        state_snapshot, "build_import_graph", lambda repo: Graph(next(edge_iter))
    )
    monkeypatch.setattr(
        state_snapshot, "SandboxSettings", lambda: SimpleNamespace(sandbox_score_db="db")
    )
    monkeypatch.setattr(
        state_snapshot, "get_latest_sandbox_score", lambda db: next(score_iter)
    )


def test_capture_and_compare(monkeypatch, tmp_path):
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0)
    (tmp_path / "mod.py").write_text("print('hi')")

    _patch_helpers(monkeypatch, [0.5, 0.6], [0.1, 0.2], [1, 2], [0.0, 0.1])

    snap1 = state_snapshot.capture_state(tmp_path, tracker)
    tracker.update(roi=2.0)
    snap2 = state_snapshot.capture_state(tmp_path, tracker)

    d = state_snapshot.compare_snapshots(snap1, snap2)

    assert pytest.approx(d["roi"], rel=1e-6) == 1.0
    assert pytest.approx(d["sandbox_score"], rel=1e-6) == 0.1
    assert pytest.approx(d["entropy"], rel=1e-6) == 0.1
    assert pytest.approx(d["call_graph_edge_count"], rel=1e-6) == 1.0
    assert pytest.approx(d["token_diversity"], rel=1e-6) == 0.1
