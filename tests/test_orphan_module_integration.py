import asyncio
import importlib.util
import json
import sys
import types
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
# provide minimal pydantic stub for optional dependencies
pyd_mod = types.ModuleType("pydantic")
sub = types.ModuleType("pydantic.dataclasses")
pyd_mod.BaseModel = type("BaseModel", (), {})
from dataclasses import dataclass as _dc
sub.dataclass = _dc
pyd_mod.dataclasses = sub
# stub jinja2 Template to avoid heavy dependency
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: object()
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)


sys.modules.setdefault("pydantic", pyd_mod)
sys.modules.setdefault("pydantic.dataclasses", sub)


# load SelfTestService from source
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"
)
sts = importlib.util.module_from_spec(spec)
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)

# load ModuleIndexDB
spec = importlib.util.spec_from_file_location(
    "module_index_db", ROOT / "module_index_db.py"
)
mod_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_db)

class DummyLogger:
    def __init__(self) -> None:
        self.exc: list[str] = []

    def exception(self, msg: str, exc: Exception | None = None) -> None:
        self.exc.append(msg)

    def info(self, *a, **k) -> None:
        pass

class DummyEngine:
    def __init__(self, index: mod_db.ModuleIndexDB) -> None:
        self.module_index = index
        self.module_clusters: dict[str, int] = {}
        self.logger = DummyLogger()
        self._last_map_refresh = 0.0
        self.auto_refresh_map = True
        self.meta_logger = None

    def _integrate_orphans(self, paths: Iterable[str]) -> None:
        if not self.module_index:
            return
        mods = {Path(p).name for p in paths}
        unknown = [m for m in mods if m not in self.module_clusters]
        if not unknown:
            return
        try:
            self.module_index.refresh(mods, force=True)
            grp_map = {m: self.module_index.get(m) for m in mods}
            self.module_clusters.update(grp_map)
            self.module_index.save()
            self._last_map_refresh = time.time()
            from sandbox_runner import environment as _env
            _env.generate_workflows_for_modules(sorted(mods))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan integration failed: %s", exc)

    def _refresh_module_map(self, modules: list[str] | None = None) -> None:
        if modules:
            self._integrate_orphans(modules)
            return


def test_orphan_module_mapping(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "orphan_modules.json").write_text(json.dumps(["foo.py"]))
    (tmp_path / "foo.py").write_text("def foo():\n    return 1\n")

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    generated: list[list[str]] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(list(mods))
        return [1]

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = fake_generate
    pkg = types.ModuleType("sandbox_runner")
    pkg.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    from sandbox_runner import environment as env

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups[str(1)] = 1
        self.save()

    monkeypatch.setattr(mod_db.ModuleIndexDB, "refresh", fake_refresh)

    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=engine._refresh_module_map,
    )
    svc.run_once()

    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    assert svc.results.get("orphan_failed") == 0
    passed = list(orphans)

    assert engine.module_clusters.get("foo.py") == 1
    data = json.loads(map_path.read_text())
    assert "foo.py" in data.get("modules", {})
    assert generated and generated[0] == ["foo.py"]


def test_recursive_orphan_module_mapping(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import ast, os
    path = ROOT / "sandbox_runner.py"
    src = path.read_text()
    tree = ast.parse(src)
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"discover_orphan_modules", "_discover_orphans_once"}
    ]
    from typing import List, Iterable
    mod_dict = {"ast": ast, "os": os, "json": json, "Path": Path, "List": List, "Iterable": Iterable}
    ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    code = ast.Module(body=funcs, type_ignores=[])
    exec(compile(code, str(path), "exec"), mod_dict)
    helper = types.ModuleType("sandbox_runner")
    helper.discover_orphan_modules = mod_dict["discover_orphan_modules"]
    helper._discover_orphans_once = mod_dict["_discover_orphans_once"]
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    generated: list[list[str]] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(list(mods))
        return [1]

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = fake_generate
    helper.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    from sandbox_runner import environment as env

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups[str(1)] = 1
        self.save()

    monkeypatch.setattr(mod_db.ModuleIndexDB, "refresh", fake_refresh)

    svc = sts.SelfTestService(
        include_orphans=True,
        recursive_orphans=True,
        integration_callback=engine._refresh_module_map,
    )
    svc.run_once()

    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    assert sorted(orphans) == ["a.py", "b.py"]
    passed = list(orphans)

    data = json.loads(map_path.read_text())
    assert "a.py" in data.get("modules", {})
    assert "b.py" in data.get("modules", {})
    assert generated and generated[0] == ["a.py", "b.py"]


def test_failed_orphans_not_added(tmp_path, monkeypatch):
    """Failing orphan modules should not be merged into the module map."""
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "foo.py").write_text("def foo():\n    return 1\n")

    # stub discovery so _update_orphan_modules adds foo.py to orphan list
    from tests.test_recursive_orphans import _load_methods
    _, _update_orphan_modules, _refresh_module_map = _load_methods()

    sr = types.ModuleType("sandbox_runner")
    sr.discover_orphan_modules = lambda repo: ["foo"]
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    _update_orphan_modules(engine)

    # module map should remain unchanged after discovery
    data = json.loads(map_path.read_text())
    assert "foo.py" not in data.get("modules", {})

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 1}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=engine._refresh_module_map,
    )
    svc.run_once()

    passed = svc.results.get("orphan_passed") or []

    data = json.loads(map_path.read_text())
    assert "foo.py" not in data.get("modules", {})
