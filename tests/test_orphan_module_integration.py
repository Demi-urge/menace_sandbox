import asyncio
import importlib.util
import json
import sys
import types
import time
from pathlib import Path
from typing import Iterable
from prometheus_client import REGISTRY

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

# reset prometheus registry to avoid duplicate metric errors when importing
REGISTRY._names_to_collectors.clear()

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

    def _collect_recursive_modules(self, modules: Iterable[str]) -> set[str]:
        return set(modules)

    def _refresh_module_map(self, modules: Iterable[str] | None = None) -> None:
        if not modules:
            return
        mods = self._collect_recursive_modules(modules)
        try:
            self.module_index.refresh(mods, force=True)
            grp_map = {m: self.module_index.get(m) for m in mods}
            self.module_clusters.update(grp_map)
            self.module_index.save()
            from sandbox_runner import environment as _env
            _env.generate_workflows_for_modules(sorted(mods))
            _env.try_integrate_into_workflows(sorted(mods))
            _env.run_workflow_simulations()
            data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
            orphan_path = data_dir / "orphan_modules.json"
            if orphan_path.exists():
                try:
                    existing = json.loads(orphan_path.read_text()) or []
                except Exception:  # pragma: no cover - best effort
                    existing = []
                keep = [p for p in existing if Path(p).name not in {Path(m).name for m in mods}]
                orphan_path.write_text(json.dumps(sorted(keep), indent=2))
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan integration failed: %s", exc)


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
    integrated: list[list[str]] = []
    ran: list[bool] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(list(mods))
        return [1]

    def fake_try(mods):
        integrated.append(list(mods))

    def fake_run():
        ran.append(True)

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = fake_generate
    env_mod.try_integrate_into_workflows = fake_try
    env_mod.run_workflow_simulations = fake_run
    pkg = types.ModuleType("sandbox_runner")
    pkg.environment = env_mod
    pkg.discover_recursive_orphans = lambda repo, module_map=None: {}
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
    assert integrated and integrated[0] == ["foo.py"]
    assert ran


def test_module_refresh_runs_simulation(tmp_path, monkeypatch):
    from tests.test_recursive_orphans import _load_methods

    _integrate_orphans, _, _refresh_module_map, _test_orphan_modules = _load_methods()

    generated: list[list[str]] = []
    integrated: list[list[str]] = []
    ran: list[bool] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(sorted(mods))
        return [1]

    def fake_try(mods):
        integrated.append(sorted(mods))

    def fake_run():
        ran.append(True)

    g = _integrate_orphans.__globals__
    g["generate_workflows_for_modules"] = fake_generate
    g["try_integrate_into_workflows"] = fake_try
    g["run_workflow_simulations"] = fake_run

    class DummyIndex:
        def __init__(self) -> None:
            self._map: dict[str, int] = {}
            self._groups: dict[str, int] = {}

        def refresh(self, modules=None, force=False):
            for m in modules or []:
                self._map[m] = 1
                self._groups.setdefault("1", 1)

        def get(self, name):
            return 1

        def save(self):
            pass

    eng = types.SimpleNamespace(
        module_index=DummyIndex(),
        module_clusters={},
        logger=DummyLogger(),
    )

    eng._collect_recursive_modules = lambda mods: {"foo.py", "dep.py"}
    eng._integrate_orphans = types.MethodType(_integrate_orphans, eng)
    eng._test_orphan_modules = types.MethodType(_test_orphan_modules, eng)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = (
        lambda repo_path, modules=None, return_details=False, **k: (
            (
                types.SimpleNamespace(
                    module_deltas={m: [1.0] for m in (modules or [])},
                    metrics_history={"synergy_roi": [0.0]},
                ),
                {m: {"sec": [{"result": {"exit_code": 0}}]} for m in modules or []},
            )
            if return_details
            else types.SimpleNamespace(
                module_deltas={m: [1.0] for m in (modules or [])},
                metrics_history={"synergy_roi": [0.0]},
            )
        )
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    _refresh_module_map(eng, ["foo.py"])

    assert generated and generated[0] == ["dep.py", "foo.py"]
    assert integrated and integrated[0] == ["dep.py", "foo.py"]
    assert ran


def test_orphan_cleanup(tmp_path, monkeypatch):
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

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = lambda mods, workflows_db="workflows.db": [1]
    env_mod.try_integrate_into_workflows = lambda mods: None
    env_mod.run_workflow_simulations = lambda: None
    pkg = types.ModuleType("sandbox_runner")
    pkg.environment = env_mod
    pkg.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    monkeypatch.setattr(mod_db.ModuleIndexDB, "refresh", lambda self, modules=None, force=False: None)

    calls: list[list[str]] = []

    def integration(mods: list[str]) -> None:
        calls.append(list(mods))
        engine._refresh_module_map(mods)

    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=integration,
        clean_orphans=True,
    )
    svc.run_once()

    data = json.loads((data_dir / "orphan_modules.json").read_text())
    assert data == []
    assert calls == [["foo.py"]]
    assert svc.results.get("orphan_passed") == ["foo.py"]


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
    orphan_path = ROOT / "sandbox_runner" / "orphan_discovery.py"
    src = orphan_path.read_text()
    tree = ast.parse(src)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "discover_orphan_modules"]

    runner_path = ROOT / "sandbox_runner.py"
    src2 = runner_path.read_text()
    tree2 = ast.parse(src2)
    for node in tree2.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_discover_orphans_once":
            funcs.append(node)
            break

    from typing import List, Iterable
    mod_dict = {"ast": ast, "os": os, "json": json, "Path": Path, "List": List, "Iterable": Iterable}
    for f in funcs:
        ast.fix_missing_locations(f)
    code = ast.Module(body=funcs, type_ignores=[])
    exec(compile(code, str(orphan_path), "exec"), mod_dict)
    helper = types.ModuleType("sandbox_runner")
    helper.discover_orphan_modules = mod_dict["discover_orphan_modules"]
    if "_discover_orphans_once" in mod_dict:
        helper._discover_orphans_once = mod_dict["_discover_orphans_once"]
    else:  # pragma: no cover - fallback if function missing
        helper._discover_orphans_once = lambda *a, **k: None
    helper.discover_recursive_orphans = (
        lambda repo, module_map=None: {"a": [], "b": []}
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    generated: list[list[str]] = []
    integrated: list[list[str]] = []
    ran: list[bool] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(list(mods))
        return [1]

    def fake_try(mods):
        integrated.append(list(mods))

    def fake_run():
        ran.append(True)

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = fake_generate
    env_mod.try_integrate_into_workflows = fake_try
    env_mod.run_workflow_simulations = fake_run
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
    assert integrated and integrated[0] == ["a.py", "b.py"]
    assert ran


def test_failed_orphans_not_added(tmp_path, monkeypatch):
    """Failing orphan modules should not be merged into the module map."""
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "foo.py").write_text("def foo():\n    return 1\n")
    (data_dir / "orphan_modules.json").write_text(json.dumps(["foo.py"]))

    from tests.test_recursive_orphans import _load_methods
    _integrate_orphans, _, _refresh_module_map, _test_orphan_modules = _load_methods()

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._test_orphan_modules = types.MethodType(_test_orphan_modules, engine)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    sr = types.ModuleType("sandbox_runner")

    def fake_run(repo_path, modules=None, return_details=False, **k):
        details = {
            m: {"sec": [{"result": {"exit_code": 1}}]} for m in modules or []
        }
        tracker = types.SimpleNamespace(
            module_deltas={m: [1.0] for m in (modules or [])},
            metrics_history={"synergy_roi": [0.0]},
        )
        return (tracker, details) if return_details else tracker

    sr.run_repo_section_simulations = fake_run
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)
    monkeypatch.chdir(tmp_path)

    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=engine._refresh_module_map,
    )
    svc.run_once()

    passed = svc.results.get("orphan_passed") or []

    data = json.loads(map_path.read_text())
    assert "foo.py" not in data.get("modules", {})
    assert passed == []
