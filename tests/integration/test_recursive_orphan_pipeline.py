from pathlib import Path
import types
import sys
import importlib

# The repository contains both a ``sandbox_runner`` package and a module of the
# same name.  To import submodules like ``sandbox_runner.cycle`` we create an
# explicit package entry with the correct ``__path__``.
pkg_path = Path(__file__).resolve().parents[2] / "sandbox_runner"
pkg = types.ModuleType("sandbox_runner")
pkg.__path__ = [str(pkg_path)]
sys.modules["sandbox_runner"] = pkg

# Stub heavy or package-relative modules required at import time.
ar_mod = types.ModuleType("adaptive_roi_predictor")
ar_mod.load_training_data = lambda *a, **k: None
sys.modules["adaptive_roi_predictor"] = ar_mod

analytics_mod = types.ModuleType("analytics")
analytics_mod.adaptive_roi_model = lambda *a, **k: None
sys.modules["analytics"] = analytics_mod

env_mod = types.ModuleType("sandbox_runner.environment")
env_mod.SANDBOX_ENV_PRESETS = []
env_mod.auto_include_modules = lambda mods, recursive=False, validate=False: (None, {"added": list(mods), "failed": [], "redundant": []})
env_mod.record_error = lambda exc: None
env_mod.run_scenarios = lambda *a, **k: None
env_mod.ERROR_CATEGORY_COUNTS = {}
env_mod.generate_workflows_for_modules = lambda mods, workflows_db="workflows.db": []
env_mod.try_integrate_into_workflows = lambda mods: None
env_mod.run_workflow_simulations = lambda: []
sys.modules["sandbox_runner.environment"] = env_mod

od_mod = types.ModuleType("sandbox_runner.orphan_discovery")
od_mod.discover_recursive_orphans = lambda path: {}
od_mod.append_orphan_cache = lambda *a, **k: None
od_mod.append_orphan_classifications = lambda *a, **k: None
od_mod.prune_orphan_cache = lambda *a, **k: None
od_mod.load_orphan_cache = lambda *a, **k: {}
od_mod.load_orphan_traces = lambda *a, **k: {}
od_mod.append_orphan_traces = lambda *a, **k: None
sys.modules["sandbox_runner.orphan_discovery"] = od_mod

rt_mod = types.ModuleType("sandbox_runner.resource_tuner")
rt_mod.ResourceTuner = lambda: types.SimpleNamespace(adjust=lambda tracker, presets: presets)
sys.modules["sandbox_runner.resource_tuner"] = rt_mod

orphan_analyzer_mod = types.ModuleType("orphan_analyzer")
orphan_analyzer_mod.analyze_redundancy = lambda p: False
sys.modules["orphan_analyzer"] = orphan_analyzer_mod

module_graph_analyzer_mod = types.ModuleType("module_graph_analyzer")
module_graph_analyzer_mod.build_import_graph = lambda repo: types.SimpleNamespace(subgraph=lambda keys: types.SimpleNamespace(copy=lambda: None))
module_graph_analyzer_mod.cluster_modules = lambda graph: {}
sys.modules["module_graph_analyzer"] = module_graph_analyzer_mod

cycle = importlib.import_module("sandbox_runner.cycle")
env = sys.modules["sandbox_runner.environment"]


def test_recursive_orphan_pipeline(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "main.py").write_text("import helper\n")  # path-ignore
    (repo / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()

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

    discover_called = {}

    def fake_discover(path):
        discover_called["called"] = True
        assert Path(path) == repo
        return {"helper": {"parents": ["main"], "redundant": False}}

    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover)

    scheduled: list[list[str]] = []
    integrated: list[str] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        scheduled.append(sorted(mods))
        return []

    def fake_integrate(mods):
        integrated.extend(sorted(mods))

    monkeypatch.setattr(env, "generate_workflows_for_modules", fake_generate)
    monkeypatch.setattr(env, "try_integrate_into_workflows", fake_integrate)
    monkeypatch.setattr(env, "run_workflow_simulations", lambda: [])

    auto_calls: list[list[str]] = []

    def fake_auto_include(mods, recursive=False, validate=False):
        auto_calls.append(sorted(mods))
        fake_generate(mods)
        fake_integrate(mods)
        return object(), {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)

    grapher_calls: list[list[str]] = []

    class DummyGrapher:
        def __init__(self, *a, **k):
            pass

        def load(self, path):
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

        def update_modules(self, paths):
            self.index_modules(paths)

        def index_modules(self, paths):
            cluster_calls.extend(str(p) for p in paths)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    cycle.include_orphan_modules(ctx)

    assert discover_called.get("called")
    assert auto_calls == [["helper.py"]]  # path-ignore
    assert grapher_calls == [["helper.py"]]  # path-ignore
    assert str(repo / "helper.py") in cluster_calls  # path-ignore
    assert scheduled == [["helper.py"]]  # path-ignore
    assert integrated == ["helper.py"]  # path-ignore
    assert ctx.module_map == {"main.py", "helper.py"}  # path-ignore
