import json
import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from menace_sandbox.roi_tracker import ROITracker


def test_include_orphan_module_integration(monkeypatch, tmp_path):
    # use light imports for sandbox_runner and stub heavy modules
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    def fake_discover(_repo):
        return {"dummy_orphan": {"classification": "candidate", "parents": []}}

    od_mod = importlib.import_module("sandbox_runner.orphan_discovery")
    od_mod.discover_recursive_orphans = fake_discover
    sys.modules["sandbox_runner.orphan_discovery"] = od_mod

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.SANDBOX_ENV_PRESETS = []

    def fake_auto_include(mods, recursive=False, validate=False):
        tracker = ROITracker()
        tracker.module_deltas = {m: [0.5] for m in mods}
        return tracker, {"added": list(mods), "failed": [], "redundant": []}

    env_mod.auto_include_modules = fake_auto_include
    sys.modules["sandbox_runner.environment"] = env_mod

    cycle = importlib.import_module("sandbox_runner.cycle")

    repo = tmp_path
    (repo / "dummy_orphan.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    orphan_file = data_dir / "orphan_modules.json"
    orphan_file.write_text(json.dumps({"dummy_orphan.py": {"classification": "candidate"}}))  # path-ignore
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    class Settings:
        auto_include_isolated = True
        recursive_isolated = True
        test_redundant_modules = False

    @dataclass
    class Ctx:
        repo: Path
        settings: Settings
        module_map: set[str] = field(default_factory=set)
        orphan_traces: dict[str, dict[str, object]] = field(default_factory=dict)
        tracker: ROITracker = field(default_factory=ROITracker)

    ctx = Ctx(repo, Settings())

    import orphan_analyzer
    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda path: False)

    class DummyMetric:
        def inc(self, *_):
            pass

        def dec(self, *_):
            pass

    monkeypatch.setattr(cycle, "orphan_modules_tested_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_reintroduced_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_failed_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_redundant_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_legacy_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_reclassified_total", DummyMetric())

    cycle.include_orphan_modules(ctx)

    assert "dummy_orphan.py" in ctx.module_map  # path-ignore
    cache_data = json.loads(orphan_file.read_text())
    assert "dummy_orphan.py" not in cache_data  # path-ignore

    traces_path = data_dir / "orphan_traces.json"
    traces = json.loads(traces_path.read_text())
    assert traces["dummy_orphan.py"]["classification_history"][-1] == "candidate"  # path-ignore
    assert traces["dummy_orphan.py"]["roi_history"] == [0.5]  # path-ignore
