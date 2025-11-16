import os
import sys
import types
import importlib
from pathlib import Path


def test_workflow_synthesis_orphan_indexing(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "main.py").write_text("import helper\n")  # path-ignore
    (repo / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()

    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

    analytics_mod = types.ModuleType("analytics")
    analytics_mod.adaptive_roi_model = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "analytics", analytics_mod)

    ar_mod = types.ModuleType("adaptive_roi_predictor")
    ar_mod.load_training_data = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "adaptive_roi_predictor", ar_mod)

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.SANDBOX_ENV_PRESETS = []
    env_mod.auto_include_modules = (
        lambda mods, recursive=False, validate=False: (None, {"added": list(mods), "failed": [], "redundant": []})
    )
    env_mod.record_error = lambda exc: None
    env_mod.run_scenarios = lambda *a, **k: None
    env_mod.ERROR_CATEGORY_COUNTS = {}
    env_mod.try_integrate_into_workflows = lambda mods, **k: None
    env_mod.generate_workflows_for_modules = lambda mods, workflows_db="workflows.db": []
    env_mod.run_workflow_simulations = lambda: []
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    discover_called = {}

    def fake_discover(path):
        discover_called["called"] = True
        assert Path(path) == repo
        return {"helper": {"parents": ["main"], "redundant": False}}

    od_mod = types.ModuleType("sandbox_runner.orphan_discovery")
    od_mod.discover_recursive_orphans = fake_discover
    od_mod.append_orphan_cache = lambda *a, **k: None
    od_mod.append_orphan_classifications = lambda *a, **k: None
    od_mod.prune_orphan_cache = lambda *a, **k: None
    od_mod.load_orphan_cache = lambda *a, **k: {}
    od_mod.load_orphan_traces = lambda *a, **k: {}
    od_mod.append_orphan_traces = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", od_mod)

    pkg_path = Path(__file__).resolve().parents[1] / "sandbox_runner"
    pkg_stub = types.ModuleType("sandbox_runner")
    pkg_stub.__path__ = [str(pkg_path)]
    pkg_stub.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg_stub)

    cycle = importlib.import_module("sandbox_runner.cycle")

    grapher_calls: list[list[str]] = []

    class DummyGrapher:
        def __init__(self, *a, **k):
            pass

        def load(self, path):
            pass

        def build_graph(self, root):
            pass

        def update_graph(self, mods):
            grapher_calls.append(list(mods))

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    cluster_calls: list[str] = []

    class DummyClusterer:
        def __init__(self, *a, **k):
            pass

        def index_modules(self, paths):
            cluster_calls.extend(str(p) for p in paths)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    ctx = types.SimpleNamespace(
        repo=repo,
        module_map={"main.py"},  # path-ignore
        orphan_traces={},
        settings=types.SimpleNamespace(
            auto_include_isolated=True,
            recursive_isolated=True,
            test_redundant_modules=False,
        ),
        tracker=types.SimpleNamespace(merge_history=lambda *a, **k: None),
    )

    cycle.include_orphan_modules(ctx)

    assert discover_called.get("called")
    assert grapher_calls == [["helper"]]
    assert str(repo / "helper.py") in cluster_calls  # path-ignore
