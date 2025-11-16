import json
import sys
import types

from sandbox_runner.orphan_integration import integrate_and_graph_orphans


def test_partial_success_marked_for_retry(tmp_path, monkeypatch):
    repo = tmp_path
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()

    class DummyTracker:
        pass

    def fake_auto_include(paths, recursive=True, router=None):
        return DummyTracker(), {"added": list(paths), "failed": [], "redundant": []}

    def fake_try_integrate(mods, router=None):
        return [1]

    env_stub = types.SimpleNamespace(
        auto_include_modules=fake_auto_include,
        try_integrate_into_workflows=fake_try_integrate,
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_stub)

    class DummyGrapher:
        def __init__(self, root=None):
            self.graph = {}

        def build_graph(self, repo):
            return {}

        def update_graph(self, names):
            raise RuntimeError("boom")

    grapher_stub = types.SimpleNamespace(
        ModuleSynergyGrapher=DummyGrapher,
        load_graph=lambda p: {},
    )
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", grapher_stub)

    class DummyClusterer:
        def index_modules(self, mods):
            pass

    monkeypatch.setitem(
        sys.modules,
        "intent_clusterer",
        types.SimpleNamespace(IntentClusterer=DummyClusterer),
    )

    tracker, results, updated, syn_ok, cl_ok = integrate_and_graph_orphans(
        repo, modules=["a.py"]  # path-ignore
    )

    assert syn_ok is False
    assert cl_ok is True
    assert results["retry"] == [str(repo / "a.py")]  # path-ignore

    log_path = repo / "sandbox_data" / "orphan_integration.log"
    data = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert data[-1]["retry"] == [str(repo / "a.py")]  # path-ignore
