from pathlib import Path
import types
import sys
import importlib
import json

# The repository contains both a ``sandbox_runner`` package and a module of the
# same name. To import submodules like ``sandbox_runner.cycle`` we create an
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
env_mod.auto_include_modules = (
    lambda mods, recursive=False, validate=False: (
        None,
        {"added": list(mods), "failed": [], "redundant": []},
    )
)
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
module_graph_analyzer_mod.build_import_graph = lambda repo: types.SimpleNamespace(
    subgraph=lambda keys: types.SimpleNamespace(copy=lambda: None)
)
module_graph_analyzer_mod.cluster_modules = lambda graph: {}
sys.modules["module_graph_analyzer"] = module_graph_analyzer_mod

cycle = importlib.import_module("sandbox_runner.cycle")


def test_orphan_cluster_update(tmp_path, monkeypatch):
    repo = tmp_path
    # Dummy workflow importing a helper module
    (repo / "workflow_wf.py").write_text("import helper_mod\n")  # path-ignore
    (repo / "helper_mod.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()

    ctx = types.SimpleNamespace(
        repo=repo,
        module_map={"workflow_wf.py"},  # path-ignore
        orphan_traces={},
        settings=types.SimpleNamespace(
            auto_include_isolated=True,
            recursive_isolated=True,
            test_redundant_modules=False,
        ),
        tracker=types.SimpleNamespace(merge_history=lambda *a, **k: None),
    )

    discover_called: dict[str, bool] = {}

    def fake_discover(path):
        discover_called["called"] = True
        assert Path(path) == repo
        return {"helper_mod": {"parents": ["workflow_wf"], "redundant": False}}

    auto_calls: dict[str, list[str] | bool] = {}

    def fake_auto(mods, recursive=False, validate=False, router=None):
        auto_calls["mods"] = list(mods)
        auto_calls["recursive"] = recursive
        return object(), {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover)
    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto)

    class DummyGrapher:
        def __init__(self, *a, root=None, **k):
            self.root = Path(root) if root else Path.cwd()
            self.graph_path: Path | None = None

        def load(self, path):
            self.graph_path = Path(path)
            return {}

        def build_graph(self, repo):
            self.graph_path = Path(repo) / "sandbox_data" / "module_synergy_graph.json"
            return {}

        def update_graph(self, mods):
            assert self.graph_path is not None
            self.graph_path.write_text(json.dumps({"nodes": list(mods)}))

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    mg_mod.load_graph = lambda path: {}
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    class DummyClusterer:
        def __init__(self, *a, local_db_path=None, shared_db_path=None, **k):
            path = Path(local_db_path or "sandbox_data/intent.db")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path = path
            self.path.write_text("[]")

        def index_modules(self, paths):
            data = json.loads(self.path.read_text())
            data.extend(str(p) for p in paths)
            self.path.write_text(json.dumps(data))

        def _load_synergy_groups(self, repo):
            return {}

        def _index_clusters(self, groups):
            pass

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    monkeypatch.chdir(repo)
    cycle.include_orphan_modules(ctx)

    assert discover_called.get("called"), "discover_recursive_orphans not called"
    assert auto_calls.get("mods") == ["helper_mod.py"]  # path-ignore
    assert auto_calls.get("recursive") is True

    graph_path = repo / "sandbox_data" / "module_synergy_graph.json"
    data = json.loads(graph_path.read_text())
    assert "helper_mod" in data.get("nodes", [])

    db_data = json.loads((repo / "sandbox_data" / "intent.db").read_text())
    assert any(Path(p).name == "helper_mod.py" for p in db_data)  # path-ignore
