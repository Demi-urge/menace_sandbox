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

    def fake_auto_include(paths, recursive=True, router=None, context_builder=None):
        return DummyTracker(), {"added": list(paths), "failed": [], "redundant": []}

    def fake_try_integrate(mods, router=None, context_builder=None):
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

    from sandbox_runner import orphan_integration as oi

    monkeypatch.setattr(oi, "resolve_path", lambda p: repo / p)
    monkeypatch.setattr(oi, "resolve_module_path", lambda p: repo / p)

    tracker, results, updated, syn_ok, cl_ok = integrate_and_graph_orphans(
        repo, modules=["a.py"], context_builder=None  # path-ignore
    )

    assert syn_ok is False
    assert cl_ok is True
    assert results["retry"] == [str(repo / "a.py")]  # path-ignore

    log_path = repo / "sandbox_data" / "orphan_integration.log"
    data = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert data[-1]["retry"] == [str(repo / "a.py")]  # path-ignore


def test_recursive_flag_forwarded(tmp_path, monkeypatch):
    repo = tmp_path
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()

    class DummyTracker:
        pass

    seen = {}

    def fake_auto_include(paths, recursive=True, router=None, context_builder=None):
        seen["recursive"] = recursive
        return DummyTracker(), {"added": list(paths), "failed": [], "redundant": []}

    def fake_try_integrate(mods, router=None, context_builder=None):
        return []

    env_stub = types.SimpleNamespace(
        auto_include_modules=fake_auto_include,
        try_integrate_into_workflows=fake_try_integrate,
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_stub)

    def resolve_stub(path):
        return repo / path

    from sandbox_runner import orphan_integration as oi

    monkeypatch.setattr(oi, "resolve_path", resolve_stub)
    monkeypatch.setattr(oi, "resolve_module_path", resolve_stub)

    grapher_stub = types.SimpleNamespace(
        ModuleSynergyGrapher=lambda root=None: types.SimpleNamespace(
            graph={}, build_graph=lambda _r: {}, update_graph=lambda _n: None
        ),
        load_graph=lambda p: {},
    )
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", grapher_stub)

    monkeypatch.setitem(
        sys.modules,
        "intent_clusterer",
        types.SimpleNamespace(IntentClusterer=lambda: types.SimpleNamespace(index_modules=lambda _m: None)),
    )

    tracker, results, updated, syn_ok, cl_ok = integrate_and_graph_orphans(
        repo,
        modules=["a.py"],
        recursive=False,
        router=None,
        context_builder=None,
    )

    assert isinstance(tracker, DummyTracker)
    assert results["added"] == [str(repo / "a.py")]
    assert updated == []
    assert syn_ok is True
    assert cl_ok is True
    assert seen.get("recursive") is False
