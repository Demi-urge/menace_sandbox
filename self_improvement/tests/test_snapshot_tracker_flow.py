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

from dynamic_path_router import resolve_path

tracker_mod = importlib.import_module(
    "menace_sandbox.self_improvement.snapshot_tracker",
)


def _patch_helpers(monkeypatch, tmp_path, entropies, diversities, complexities):
    ent_iter = iter(entropies)
    div_iter = iter(diversities)
    comp_iter = iter(complexities)

    monkeypatch.setattr(
        tracker_mod,
        "collect_snapshot_metrics",
        lambda files, settings=None: (next(ent_iter), next(div_iter)),
    )
    monkeypatch.setattr(
        tracker_mod, "compute_call_graph_complexity", lambda repo: next(comp_iter)
    )
    monkeypatch.setattr(tracker_mod, "relevancy_radar", None)
    monkeypatch.setattr(
        tracker_mod,
        "SandboxSettings",
        lambda: SimpleNamespace(
            snapshot_metrics={
                "roi",
                "sandbox_score",
                "entropy",
                "call_graph_complexity",
                "token_diversity",
            },
            sandbox_data_dir=str(tmp_path),
            sandbox_repo_path=str(tmp_path),
        ),
    )


def test_capture_and_compare(monkeypatch, tmp_path):
    (tmp_path / resolve_path("mod.py")).write_text("print('hi')")

    _patch_helpers(monkeypatch, tmp_path, [0.5, 0.6], [0.1, 0.2], [1, 2])

    snap1 = tracker_mod.capture(
        stage="pre",
        files=[tmp_path / resolve_path("mod.py")],
        roi=1.0,
        sandbox_score=0.0,
    )
    snap2 = tracker_mod.capture(
        stage="post",
        files=[tmp_path / resolve_path("mod.py")],
        roi=2.0,
        sandbox_score=0.1,
    )

    d = tracker_mod.compute_delta(snap1, snap2)

    assert pytest.approx(d["roi"], rel=1e-6) == 1.0
    assert pytest.approx(d["sandbox_score"], rel=1e-6) == 0.1
    assert pytest.approx(d["entropy"], rel=1e-6) == 0.1
    assert pytest.approx(d["call_graph_complexity"], rel=1e-6) == 1.0
    assert pytest.approx(d["token_diversity"], rel=1e-6) == 0.1

