import json
import sys
import types
import warnings
import importlib
import shutil
from pathlib import Path

import pytest

from menace_sandbox.foresight_tracker import ForesightTracker

# Ensure sandbox imports see a minimal menace package
sys.modules.setdefault("menace", types.ModuleType("menace")).RAISE_ERRORS = False

adp = types.ModuleType("adaptive_roi_predictor")
adp.load_training_data = lambda *a, **k: []
sys.modules.setdefault("adaptive_roi_predictor", adp)

env_stub = sys.modules.setdefault(
    "sandbox_runner.environment", types.ModuleType("sandbox_runner.environment")
)
env_stub.record_error = lambda *a, **k: None
env_stub.SANDBOX_ENV_PRESETS = [{}]
env_stub.load_presets = lambda: env_stub.SANDBOX_ENV_PRESETS
env_stub.run_scenarios = lambda *a, **k: None
env_stub.ERROR_CATEGORY_COUNTS = {}
env_stub.auto_include_modules = lambda *a, **k: []

from sandbox_runner import bootstrap
import sandbox_runner.generative_stub_provider as gsp
from dynamic_path_router import resolve_path


class DummyROITracker:
    def __init__(self, deltas):
        self._deltas = iter(deltas)
        self.raroi_history = [0.0]
        self.confidence_history = [0.0]
        self.metrics_history = {"synergy_resilience": [0.0]}

    def next_delta(self):
        delta = next(self._deltas)
        self.raroi_history.append(self.raroi_history[-1] + delta / 2.0)
        return delta

    def scenario_degradation(self):  # pragma: no cover - deterministic
        return 0.0


class MiniSelfImprovementEngine:
    def __init__(self, tracker, foresight_tracker):
        self.tracker = tracker
        self.foresight_tracker = foresight_tracker

    def run_cycle(self, workflow_id="wf"):
        delta = self.tracker.next_delta()
        raroi_delta = self.tracker.raroi_history[-1] - self.tracker.raroi_history[-2]
        confidence = self.tracker.confidence_history[-1]
        resilience = self.tracker.metrics_history["synergy_resilience"][-1]
        scenario_deg = self.tracker.scenario_degradation()
        self.foresight_tracker.record_cycle_metrics(
            workflow_id,
            {
                "roi_delta": float(delta),
                "raroi_delta": float(raroi_delta),
                "confidence": float(confidence),
                "resilience": float(resilience),
                "scenario_degradation": float(scenario_deg),
            },
        )


def sample_func(name: str, count: int) -> None:  # pragma: no cover - helper
    return None


def test_full_initialisation_and_cycle_with_stub(tmp_path, monkeypatch):
    """Launch sandbox, generate stubs and run a self-improvement cycle."""
    for name in ("relevancy_radar", "quick_fix_engine"):
        mod = types.ModuleType(name)
        mod.__version__ = "1.0.0"
        monkeypatch.setitem(sys.modules, name, mod)

    async def fake_aload_generator():
        return None

    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)
    cache_file = tmp_path / "stub_cache.json"
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(cache_file))
    gsp._CACHE.clear()
    gsp._CONFIG = None
    gsp._SETTINGS = None

    events = []

    def fake_main(_args):
        ft = ForesightTracker(max_cycles=1)
        tracker = DummyROITracker([1.0])
        stub = gsp.generate_stubs(
            [{}], {"target": sample_func}, context_builder=types.SimpleNamespace(build_prompt=lambda q, *, intent_metadata=None, **k: q)
        )[0]
        gsp._save_cache()
        events.append(stub["count"])
        engine = MiniSelfImprovementEngine(tracker, ft)
        engine.run_cycle()
        events.append(ft.history["wf"][0]["roi_delta"])

    monkeypatch.setattr(bootstrap, "_cli_main", fake_main)
    sys.modules.pop("sandbox_settings", None)
    SandboxSettings = importlib.import_module("sandbox_settings").SandboxSettings
    settings = SandboxSettings(
        sandbox_data_dir=str(tmp_path), menace_env_file=str(tmp_path / ".env")
    )
    bootstrap.launch_sandbox(settings, verifier=lambda s: None)
    assert events[1] == 1.0
    assert isinstance(events[0], int)
    assert cache_file.exists()


def test_dependency_check_raises_on_missing(monkeypatch):
    """_verify_required_dependencies exits when tools or packages missing."""
    class DummySettings:
        required_system_tools = ["missing_tool"]
        required_python_packages = ["missing_pkg"]
        optional_python_packages: list[str] | None = []

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner_main",
        resolve_path("sandbox_runner.py"),  # path-ignore
    )
    sr = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with pytest.raises(SystemExit):
        spec.loader.exec_module(sr)


def test_stub_generation_handles_corrupt_cache(tmp_path, monkeypatch):
    """Corrupted cache triggers warning and backup file creation."""
    cache_file = tmp_path / "stub_cache.json"
    cache_file.write_text("not-json", encoding="utf-8")
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(cache_file))
    gsp._CACHE.clear()
    gsp._CONFIG = None
    gsp._SETTINGS = None
    async def fake_aload_generator():
        return None
    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)
    builder = types.SimpleNamespace(build_prompt=lambda q, *, intent_metadata=None, **k: q)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stub = gsp.generate_stubs([{}], {"target": sample_func}, context_builder=builder)[0]
    assert any(isinstance(item.message, gsp.StubCacheWarning) for item in w)
    assert cache_file.with_suffix(".corrupt").exists()
    assert gsp._type_matches(stub["count"], int)


def test_stub_generation_handles_invalid_entry(tmp_path, monkeypatch):
    """Invalid cache entries are ignored with a warning."""
    cache_file = tmp_path / "stub_cache.json"
    bad_entry = [["mod::func", [1, 2, 3]]]
    cache_file.write_text(json.dumps(bad_entry), encoding="utf-8")
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(cache_file))
    gsp._CACHE.clear()
    gsp._CONFIG = None
    gsp._SETTINGS = None
    async def fake_aload_generator():
        return None
    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)
    builder = types.SimpleNamespace(build_prompt=lambda q, *, intent_metadata=None, **k: q)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stub = gsp.generate_stubs([{}], {"target": sample_func}, context_builder=builder)[0]
    assert any(isinstance(item.message, gsp.StubCacheWarning) for item in w)
    assert isinstance(stub["count"], int)
