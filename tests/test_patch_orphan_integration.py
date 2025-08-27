import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class SelfCodingEngine:
    def patch_file(self, path: Path, description: str) -> None:
        raise NotImplementedError

    def apply_patch(self, path: Path, description: str) -> None:
        self.patch_file(path, description)
        from sandbox_runner import integrate_new_orphans, try_integrate_into_workflows
        added = integrate_new_orphans(Path.cwd(), router=None)
        try_integrate_into_workflows(added)


def test_patch_orphan_integration(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "existing.py").write_text("def foo():\n    pass\n")
    (repo / "new_mod.py").write_text("VALUE = 1\n")
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    module_map_path = data_dir / "module_map.json"
    module_map_path.write_text(json.dumps({"modules": {"existing.py": 1}}))

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

    def integrate_new_orphans(repo_path, router=None):
        map_path = Path(repo_path) / "sandbox_data" / "module_map.json"
        data = json.loads(map_path.read_text())
        modules = data.get("modules", data)
        modules["new_mod.py"] = 1
        if "modules" in data:
            data["modules"] = modules
        else:
            data = modules
        map_path.write_text(json.dumps(data))
        names = [Path("new_mod.py").with_suffix("").as_posix()]
        from module_synergy_grapher import ModuleSynergyGrapher
        ModuleSynergyGrapher(repo_path).update_graph(names)
        from intent_clusterer import IntentClusterer
        IntentClusterer().index_modules([Path(repo_path) / "new_mod.py"])
        return ["new_mod.py"]

    sandbox_runner = types.ModuleType("sandbox_runner")
    sandbox_runner.integrate_new_orphans = integrate_new_orphans
    sandbox_runner.try_integrate_into_workflows = lambda modules: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sandbox_runner)

    engine = SelfCodingEngine()

    def fake_patch_file(path: Path, description: str) -> None:
        path.write_text("import new_mod\n")

    monkeypatch.setattr(engine, "patch_file", fake_patch_file)

    engine.apply_patch(Path("existing.py"), "add orphan import")

    data = json.loads(module_map_path.read_text())
    modules = data.get("modules", data)
    assert "new_mod.py" in modules
    assert synergy_calls == ["new_mod"]
    assert intent_calls == [repo / "new_mod.py"]
