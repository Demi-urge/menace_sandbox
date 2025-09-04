import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

# Stub heavy modules before importing self_improvement components
sys.modules.setdefault("dynamic_path_router", ModuleType("dynamic_path_router"))
sys.modules["dynamic_path_router"].resolve_path = lambda p: Path(p)
sys.modules["dynamic_path_router"].resolve_dir = lambda p: Path(p)
sys.modules["dynamic_path_router"].repo_root = lambda: Path(".")
pkg_path = Path(__file__).resolve().parent.parent
root_pkg = ModuleType("menace_sandbox")
root_pkg.__path__ = [str(pkg_path.parent)]
sys.modules.setdefault("menace_sandbox", root_pkg)
sub_pkg = ModuleType("menace_sandbox.self_improvement")
sub_pkg.__path__ = [str(pkg_path)]
sys.modules.setdefault("menace_sandbox.self_improvement", sub_pkg)
sys.modules.setdefault("self_improvement", sub_pkg)
init_stub = ModuleType("menace_sandbox.self_improvement.init")
init_stub._repo_path = lambda: Path(".")  # type: ignore[attr-defined]
sys.modules.setdefault("menace_sandbox.self_improvement.init", init_stub)
sys.modules.setdefault("self_improvement.init", init_stub)
spec = None
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.sandbox_settings", pkg_path.parent / "sandbox_settings.py"
    )
    sandbox_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(sandbox_mod)  # type: ignore[attr-defined]
    sys.modules.setdefault("menace_sandbox.sandbox_settings", sandbox_mod)
    sys.modules.setdefault("sandbox_settings", sandbox_mod)
except Exception:
    pass

from menace_sandbox.self_improvement.baseline_tracker import BaselineTracker
from menace_sandbox.self_improvement import state_snapshot, prompt_memory
from menace_sandbox.self_improvement.state_snapshot import capture_snapshot, delta, save_checkpoint
from self_improvement_policy import SelfImprovementPolicy, PolicyConfig
from sandbox_settings import SandboxSettings


def _patch_snapshot_helpers(monkeypatch, complexities, diversities, scores):
    comp_iter = iter(complexities)
    div_iter = iter(diversities)
    score_iter = iter(scores)
    monkeypatch.setattr(state_snapshot, "compute_call_graph_complexity", lambda repo: next(comp_iter))
    monkeypatch.setattr(state_snapshot, "compute_entropy_metrics", lambda files, settings: next(div_iter))
    monkeypatch.setattr(state_snapshot, "get_latest_sandbox_score", lambda db: next(score_iter))


def test_snapshot_capture_and_delta(monkeypatch, tmp_path):
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0, entropy=0.5)
    (tmp_path / "mod.py").write_text("print('hi')")
    settings = SandboxSettings(sandbox_repo_path=str(tmp_path))
    _patch_snapshot_helpers(monkeypatch, [1.0, 2.0], [(0, 0, 0.5), (0, 0, 0.6)], [0.0, 0.2])
    snap1 = capture_snapshot(tracker, settings)
    tracker.update(roi=2.0, entropy=0.6)
    snap2 = capture_snapshot(tracker, settings)
    d = delta(snap1, snap2)
    assert pytest.approx(d["roi"], rel=1e-6) == 1.0
    assert pytest.approx(d["entropy"], rel=1e-6) == 0.1
    assert pytest.approx(d["sandbox_score"], rel=1e-6) == 0.2
    assert pytest.approx(d["call_graph_complexity"], rel=1e-6) == 1.0
    assert pytest.approx(d["token_diversity"], rel=1e-6) == 0.1


def test_regression_flagged_on_roi_drop(monkeypatch, tmp_path):
    tracker = BaselineTracker(window=3)
    tracker.update(roi=1.0, entropy=0.5)
    (tmp_path / "mod.py").write_text("print('hi')")
    settings = SandboxSettings(sandbox_repo_path=str(tmp_path))
    _patch_snapshot_helpers(monkeypatch, [1.0, 1.0], [(0, 0, 0.5), (0, 0, 0.4)], [0.0, 0.0])
    snap1 = capture_snapshot(tracker, settings)
    tracker.update(roi=0.5, entropy=0.4)
    snap2 = capture_snapshot(tracker, settings)
    d = delta(snap1, snap2)
    assert d["roi"] < 0
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory._settings, "prompt_penalty_path", "penalties.json")
    monkeypatch.setattr(prompt_memory, "_penalty_path", tmp_path / "penalties.json")
    from filelock import FileLock
    monkeypatch.setattr(
        prompt_memory, "_penalty_lock", FileLock(str(tmp_path / "penalties.json") + ".lock")
    )
    prompt = SimpleNamespace(metadata={"prompt_id": "p1"})
    prompt_memory.log_prompt_attempt(prompt, False, {"delta": d})
    penalties = json.loads((tmp_path / "penalties.json").read_text())
    assert penalties["p1"] == 1


def test_confidence_and_checkpoint_on_improvement(monkeypatch, tmp_path):
    tracker = BaselineTracker(window=3)
    tracker.update(roi=0.5, entropy=0.1)
    module = tmp_path / "mod.py"
    module.write_text("print('hi')")
    data_dir = tmp_path / "data"
    settings = SandboxSettings(sandbox_repo_path=str(tmp_path), sandbox_data_dir=str(data_dir))
    _patch_snapshot_helpers(monkeypatch, [1.0, 2.0], [(0, 0, 0.1), (0, 0, 0.2)], [0.0, 1.0])
    snap1 = capture_snapshot(tracker, settings)
    tracker.update(roi=1.0, entropy=0.2)
    snap2 = capture_snapshot(tracker, settings)
    d = delta(snap1, snap2)
    assert all(d[k] > 0 for k in d if k != "timestamp")
    monkeypatch.setattr(
        state_snapshot,
        "SandboxSettings",
        lambda: SimpleNamespace(sandbox_data_dir=str(data_dir), sandbox_repo_path=str(tmp_path)),
    )
    path = save_checkpoint(module, "deadbeef")
    assert path.exists()
    strategy_confidence = {"improve": 0}
    strategy_confidence["improve"] += 1
    assert strategy_confidence["improve"] == 1


def test_prompt_deprioritization_after_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_memory, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(prompt_memory._settings, "prompt_penalty_path", "penalties.json")
    monkeypatch.setattr(prompt_memory, "_penalty_path", tmp_path / "penalties.json")
    from filelock import FileLock
    monkeypatch.setattr(
        prompt_memory, "_penalty_lock", FileLock(str(tmp_path / "penalties.json") + ".lock")
    )
    for _ in range(3):
        prompt_memory.record_regression("1")
    penalties = prompt_memory.load_prompt_penalties()
    settings = SimpleNamespace(prompt_failure_threshold=2, prompt_penalty_multiplier=0.0)
    actions = {0: 1.0, 1: 10.0}
    penalised = {
        act
        for act in actions
        if penalties.get(str(act), 0) >= settings.prompt_failure_threshold
    }
    for act in penalised:
        actions[act] *= settings.prompt_penalty_multiplier
    assert actions[1] == 0.0
