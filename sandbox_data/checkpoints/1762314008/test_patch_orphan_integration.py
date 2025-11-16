import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class ContextBuilder:
    pass


class SelfCodingEngine:
    def __init__(self, context_builder=None):
        self.context_builder = context_builder

    def patch_file(self, path: Path, description: str) -> None:
        raise NotImplementedError

    def apply_patch(self, path: Path, description: str) -> None:
        self.patch_file(path, description)
        from sandbox_runner.orphan_integration import integrate_orphans
        integrate_orphans(Path.cwd(), router=None)


def test_patch_orphan_integration(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "existing.py").write_text("def foo():\n    pass\n")  # path-ignore
    (repo / "new_mod.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    module_map_path = data_dir / "module_map.json"
    module_map_path.write_text(json.dumps({"modules": {"existing.py": 1}}))  # path-ignore

    monkeypatch.chdir(repo)

    synergy_calls: list[str] = []
    intent_calls: list[Path] = []

    class DummyGrapher:
        def __init__(self, root: Path):
            pass

        def update_graph(self, names):
            synergy_calls.extend(names)

    monkeypatch.setitem(
        sys.modules,
        "module_synergy_grapher",
        SimpleNamespace(ModuleSynergyGrapher=DummyGrapher, load_graph=lambda p: None),
    )

    class DummyClusterer:
        def __init__(self):
            pass

        def index_modules(self, paths):
            intent_calls.extend(paths)

    monkeypatch.setitem(
        sys.modules,
        "intent_clusterer",
        SimpleNamespace(IntentClusterer=DummyClusterer),
    )

    def integrate_orphans(repo_path, router=None):
        map_path = Path(repo_path) / "sandbox_data" / "module_map.json"
        data = json.loads(map_path.read_text())
        modules = data.get("modules", data)
        modules["new_mod.py"] = 1  # path-ignore
        if "modules" in data:
            data["modules"] = modules
        else:
            data = modules
        map_path.write_text(json.dumps(data))
        names = [Path("new_mod.py").with_suffix("").as_posix()]  # path-ignore
        from module_synergy_grapher import ModuleSynergyGrapher
        ModuleSynergyGrapher(repo_path).update_graph(names)
        from intent_clusterer import IntentClusterer
        IntentClusterer().index_modules([Path(repo_path) / "new_mod.py"])  # path-ignore
        return ["new_mod.py"]  # path-ignore

    sandbox_runner_pkg = types.ModuleType("sandbox_runner")
    sandbox_runner_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "sandbox_runner", sandbox_runner_pkg)
    oi_mod = types.ModuleType("sandbox_runner.orphan_integration")
    oi_mod.integrate_orphans = integrate_orphans
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", oi_mod)

    builder = ContextBuilder()
    engine = SelfCodingEngine(builder)

    def fake_patch_file(path: Path, description: str) -> None:
        path.write_text("import new_mod\n")

    monkeypatch.setattr(engine, "patch_file", fake_patch_file)

    engine.apply_patch(Path("existing.py"), "add orphan import")  # path-ignore

    data = json.loads(module_map_path.read_text())
    modules = data.get("modules", data)
    assert "new_mod.py" in modules  # path-ignore
    assert synergy_calls == ["new_mod"]
    assert intent_calls == [repo / "new_mod.py"]  # path-ignore
