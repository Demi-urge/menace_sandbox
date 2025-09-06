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


def test_orphan_inclusion_after_patch(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "existing.py").write_text("def foo():\n    pass\n")  # path-ignore
    (repo / "orphan.py").write_text("VALUE = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    module_map_path = data_dir / "module_map.json"
    module_map_path.write_text(json.dumps({"modules": {"existing.py": 1}}))  # path-ignore

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    intent_calls: list[Path] = []

    def integrate_orphans(repo_path, router=None):
        map_path = Path(repo_path) / "sandbox_data" / "module_map.json"
        data = json.loads(map_path.read_text())
        modules = data.get("modules", data)
        modules["orphan.py"] = 1  # path-ignore
        if "modules" in data:
            data["modules"] = modules
        else:
            data = modules
        map_path.write_text(json.dumps(data))
        graph_path = Path(repo_path) / "sandbox_data" / "module_synergy_graph.json"
        graph_path.write_text(json.dumps({"modules": ["orphan"]}))
        intent_calls.append(Path(repo_path) / "orphan.py")  # path-ignore
        return ["orphan.py"]  # path-ignore

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    oi_mod = types.ModuleType("sandbox_runner.orphan_integration")
    oi_mod.integrate_orphans = integrate_orphans
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", oi_mod)

    builder = ContextBuilder()
    engine = SelfCodingEngine(builder)

    def fake_patch_file(path: Path, description: str) -> None:
        path.write_text("import orphan\n")

    monkeypatch.setattr(engine, "patch_file", fake_patch_file)

    engine.apply_patch(Path("existing.py"), "add orphan import")  # path-ignore

    graph_path = repo / "sandbox_data" / "module_synergy_graph.json"
    data = json.loads(graph_path.read_text())
    assert "orphan" in data["modules"]
    assert intent_calls == [repo / "orphan.py"]  # path-ignore
