import json
import sys
import types
import json
import sys
import types
from pathlib import Path


class ContextBuilder:
    pass


class SelfCodingEngine:
    def __init__(self, context_builder=None):
        self.context_builder = context_builder

    def patch_file(self, path: Path, description: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def apply_patch(self, path: Path, description: str) -> None:
        self.patch_file(path, description)
        from sandbox_runner.orphan_integration import integrate_orphans
        integrate_orphans(Path.cwd())


def test_post_synthesis_orphan_inclusion(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "existing.py").write_text("def foo():\n    pass\n")  # path-ignore
    (repo / "helper.py").write_text("import new_module\n")  # path-ignore
    (repo / "new_module.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    module_map_path = data_dir / "module_map.json"
    module_map_path.write_text(json.dumps({"modules": {"existing.py": 1}}))  # path-ignore
    graph_path = data_dir / "module_synergy_graph.json"
    graph_path.write_text(json.dumps({"modules": []}))

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    grapher_updates: list[str] = []
    clusterer_calls: list[str] = []

    class DummyGrapher:
        def __init__(self, root):
            self.root = root

        def build_graph(self, repo):  # pragma: no cover - stub
            return {}

        def update_graph(self, names):
            grapher_updates.extend(names)
            data = json.loads(graph_path.read_text())
            modules = data.get("modules", [])
            modules.extend(names)
            data["modules"] = modules
            graph_path.write_text(json.dumps(data))

    class DummyClusterer:
        def __init__(self):  # pragma: no cover - stub
            pass

        def index_modules(self, paths):
            clusterer_calls.extend(paths)

    mg_mod = types.ModuleType("module_synergy_grapher")
    mg_mod.ModuleSynergyGrapher = DummyGrapher
    mg_mod.load_graph = lambda p: {"modules": []}
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", mg_mod)

    ic_mod = types.ModuleType("intent_clusterer")
    ic_mod.IntentClusterer = DummyClusterer
    monkeypatch.setitem(sys.modules, "intent_clusterer", ic_mod)

    def auto_include_modules(paths, recursive=True, router=None, context_builder=None):
        data = json.loads(module_map_path.read_text())
        modules = data.get("modules", data)
        for p in paths:
            modules[p] = 1
        data["modules"] = modules
        module_map_path.write_text(json.dumps(data))
        grapher = mg_mod.ModuleSynergyGrapher(repo)
        grapher.update_graph([Path(p).stem for p in paths])
        clusterer = ic_mod.IntentClusterer()
        clusterer.index_modules(paths)
        return None, {"added": list(paths)}

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.auto_include_modules = auto_include_modules
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    def integrate_orphans(repo_path, router=None):
        from sandbox_runner.environment import auto_include_modules
        auto_include_modules(
            ["helper.py", "new_module.py"], context_builder=None
        )  # path-ignore
        return ["helper.py", "new_module.py"]  # path-ignore

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    oi_mod = types.ModuleType("sandbox_runner.orphan_integration")
    oi_mod.integrate_orphans = integrate_orphans
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", oi_mod)

    builder = ContextBuilder()
    engine = SelfCodingEngine(builder)

    def fake_patch_file(path: Path, description: str) -> None:
        path.write_text("import helper\n")

    monkeypatch.setattr(engine, "patch_file", fake_patch_file)

    engine.apply_patch(Path("existing.py"), "add helper import")  # path-ignore

    module_map = json.loads(module_map_path.read_text())["modules"]
    assert "new_module.py" in module_map  # path-ignore
    assert "helper.py" in module_map  # path-ignore

    graph_data = json.loads(graph_path.read_text())
    assert "new_module" in graph_data["modules"]
    assert "helper" in graph_data["modules"]

    assert clusterer_calls == ["helper.py", "new_module.py"]  # path-ignore
