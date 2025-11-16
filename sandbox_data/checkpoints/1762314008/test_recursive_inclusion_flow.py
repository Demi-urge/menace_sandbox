import json
import json
import sys
import types
from pathlib import Path

import ast

import sandbox_runner.environment as env
from context_builder_util import create_context_builder


class DummyTracker:
    def __init__(self) -> None:
        self.cluster_map: dict[str, int] = {}

    def save_history(self, path: str) -> None:  # pragma: no cover - simple write
        Path(path).write_text("{}")


def test_recursive_inclusion_flow(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "iso.py").write_text("import dep\nimport red\n")  # path-ignore
    (repo / "dep.py").write_text("x = 1\n")  # path-ignore
    (repo / "red.py").write_text("x = 2\n")  # path-ignore

    data_dir = repo / "sandbox_data"
    data_dir.mkdir()

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    class Settings:
        auto_include_isolated = True
        recursive_isolated = True
        recursive_orphan_scan = True

    monkeypatch.setitem(
        sys.modules,
        "sandbox_settings",
        types.SimpleNamespace(SandboxSettings=lambda: Settings()),
    )

    def collect_deps(mods):
        mods = set(mods)
        if "iso.py" in mods:  # path-ignore
            mods.update({"dep.py", "red.py"})  # path-ignore
        return mods

    monkeypatch.setitem(
        sys.modules,
        "sandbox_runner.dependency_utils",
        types.SimpleNamespace(collect_local_dependencies=collect_deps),
    )

    monkeypatch.setitem(
        sys.modules,
        "orphan_analyzer",
        types.SimpleNamespace(
            analyze_redundancy=lambda p: Path(p).name == "red.py"  # path-ignore
        ),
    )

    iso_mod = types.SimpleNamespace(
        discover_isolated_modules=lambda repo, recursive=True: ["iso.py"]  # path-ignore
    )
    scripts_pkg = types.SimpleNamespace(discover_isolated_modules=iso_mod)
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", iso_mod)

    class STS1:
        def __init__(self, pytest_args, **kwargs):
            pass

        def run_once(self):
            return ({}, [])

    monkeypatch.setitem(
        sys.modules, "self_test_service", types.SimpleNamespace(SelfTestService=STS1)
    )

    monkeypatch.setattr(
        env,
        "generate_workflows_for_modules",
        lambda mods, workflows_db="workflows.db", context_builder=None: None,
    )
    monkeypatch.setattr(
        env, "try_integrate_into_workflows", lambda mods, context_builder=None: None
    )
    monkeypatch.setattr(
        env,
        "run_workflow_simulations",
        lambda *a, **k: DummyTracker(),
    )

    env.auto_include_modules(
        ["iso.py"],
        recursive=True,
        validate=True,
        context_builder=create_context_builder(),
    )  # path-ignore

    map_data = json.loads((data_dir / "module_map.json").read_text())
    assert set(map_data) == {"iso.py", "dep.py"}  # path-ignore
    assert "red.py" not in map_data  # path-ignore

    orphan_cache = json.loads((data_dir / "orphan_modules.json").read_text())
    assert orphan_cache == {"red.py": {"redundant": True}}  # path-ignore

    (data_dir / "orphan_modules.json").write_text(
        json.dumps(["iso.py", "dep.py", "red.py"])  # path-ignore
    )

    class STS2:
        def __init__(self, pytest_args, **kwargs):
            self.results: dict = {}

        async def _run_once(self):
            self.results = {
                "integration": {
                    "integrated": ["iso.py", "dep.py"],  # path-ignore
                    "redundant": ["red.py"],  # path-ignore
                }
            }

    monkeypatch.setitem(
        sys.modules, "self_test_service", types.SimpleNamespace(SelfTestService=STS2)
    )

    recorded: list[list[str]] = []

    def auto_stub(mods, recursive=False, validate=False, context_builder=None):
        recorded.append(sorted(mods))

    monkeypatch.setattr(env, "auto_include_modules", auto_stub)

    def _load_update_method():
        path = Path(__file__).resolve().parents[1] / "self_improvement.py"  # path-ignore
        src = path.read_text()
        tree = ast.parse(src)
        method = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_update_orphan_modules":
                        method = item
                        break
        assert method is not None
        g = {
            "Path": Path,
            "os": __import__("os"),
            "json": json,
            "SandboxSettings": Settings,
            "SelfTestService": STS2,
            "environment": env,
            "asyncio": __import__("asyncio"),
            "Iterable": __import__("typing").Iterable,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), g)
        return g["_update_orphan_modules"]

    update_fn = _load_update_method()

    class DummyIndex:
        def __init__(self):
            self.ids: dict[str, int] = {}
            self.next = 1

        def refresh(self, modules=None, force=False):
            for m in modules or []:
                if m not in self.ids:
                    self.ids[m] = self.next
                    self.next += 1

        def save(self):
            pass

        def get(self, name):
            return self.ids.get(name)

    class DummyLogger:
        def info(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

    eng = types.SimpleNamespace(
        module_index=DummyIndex(),
        module_clusters={},
        logger=DummyLogger(),
    )

    def guarded(self, modules=None):
        if getattr(self, "_guard", False):
            return
        self._guard = True
        try:
            return update_fn(self, modules)
        finally:
            self._guard = False

    eng._update_orphan_modules = types.MethodType(guarded, eng)

    eng._update_orphan_modules(["iso.py", "dep.py", "red.py"])  # path-ignore

    assert json.loads((data_dir / "orphan_modules.json").read_text()) == ["red.py"]  # path-ignore
    assert set(eng.module_clusters) == {"iso.py", "dep.py"}  # path-ignore
