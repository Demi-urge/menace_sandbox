import os
import sys
import types
import json
import asyncio
import importlib.util
import ast
import subprocess
from pathlib import Path

import pytest

# ensure lightweight imports for menace modules
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# stub heavy optional dependencies
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)

yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)

sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))

sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.discover_recursive_orphans = lambda *a, **k: {}
sandbox_runner_pkg.__path__ = []
sys.modules["sandbox_runner"] = sandbox_runner_pkg
orphan_disc_mod = types.ModuleType("sandbox_runner.orphan_discovery")
orphan_disc_mod.discover_recursive_orphans = lambda *a, **k: {}
orphan_disc_mod.append_orphan_cache = lambda *a, **k: None
orphan_disc_mod.append_orphan_classifications = lambda *a, **k: None
orphan_disc_mod.prune_orphan_cache = lambda *a, **k: None
orphan_disc_mod.load_orphan_cache = lambda *a, **k: {}
sys.modules.setdefault("sandbox_runner.orphan_discovery", orphan_disc_mod)
sys.modules.setdefault("orphan_discovery", orphan_disc_mod)

ROOT = Path(__file__).resolve().parents[1]

# dynamically load SelfTestService
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
self_test_mod = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
else:
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
    sys.modules["menace"] = pkg
from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()
spec.loader.exec_module(self_test_mod)


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}
self_test_mod.analyze_redundancy = lambda p: False

# stub ModuleIndexDB
module_index_db = types.ModuleType("module_index_db")


class ModuleIndexDB:
    def __init__(self, path: Path):
        self._map: dict[str, int] = {}
        self._groups: dict[str, int] = {}
        self._tags: dict[str, list[str]] = {}
        self.path = Path(path)

    def save(self) -> None:
        data = {"modules": self._map, "groups": self._groups}
        self.path.write_text(json.dumps(data))

    def refresh(self, modules=None, force=False):
        pass


module_index_db.ModuleIndexDB = ModuleIndexDB
sys.modules.setdefault("module_index_db", module_index_db)


def _load_refresh_methods(fake_generate, fake_try, fake_run=lambda *a, **k: None):
    """Extract refresh helpers from self_improvement with hooks."""
    path = ROOT / "self_improvement.py"  # path-ignore
    src = path.read_text()
    tree = ast.parse(src)
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "_integrate_orphans",
                    "_refresh_module_map",
                }:
                    methods.append(item)
    mod_dict = {
        "os": os,
        "json": json,
        "Path": Path,
        "time": __import__("time"),
        "datetime": __import__("datetime").datetime,
        "Iterable": __import__("typing").Iterable,
        "build_module_map": lambda repo, ignore=None: {},
        "generate_workflows_for_modules": fake_generate,
        "try_integrate_into_workflows": fake_try,
        "run_workflow_simulations": fake_run,
        "log_record": lambda **k: {},
        "analyze_redundancy": lambda p: False,
    }
    def auto_include_modules(mods, recursive=False, context_builder=None):
        fake_generate(mods)
        fake_try(mods)
        fake_run()
    mod_dict["auto_include_modules"] = auto_include_modules
    ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    code = ast.Module(body=methods, type_ignores=[])
    exec(compile(code, str(path), "exec"), mod_dict)
    return mod_dict["_integrate_orphans"], mod_dict["_refresh_module_map"]


@pytest.mark.skip(reason="Legacy integration path disabled in test environment")
def test_recursive_isolated_integration(monkeypatch, tmp_path):
    # prepare small repo tree
    (tmp_path / "sandbox_data").mkdir()
    (tmp_path / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "isolated.py").write_text("import helper\n")  # path-ignore

    # stub discovery to ensure recursion
    called: dict[str, object] = {}

    def fake_discover(root, *, recursive=True):
        called["root"] = Path(root)
        called["recursive"] = recursive
        mods = ["isolated.py"]  # path-ignore
        if recursive:
            mods.append("helper.py")  # path-ignore
        return mods

    iso_mod = types.ModuleType("scripts.discover_isolated_modules")
    iso_mod.discover_isolated_modules = fake_discover
    pkg_scripts = types.ModuleType("scripts")
    pkg_scripts.discover_isolated_modules = iso_mod
    monkeypatch.setitem(sys.modules, "scripts", pkg_scripts)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", iso_mod)

    # capture executed paths
    executed: list[str] = []

    async def fake_exec(*cmd, **kwargs):
        mod_arg = None
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.endswith(".py") and mod_arg is None:  # path-ignore
                mod_arg = s
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
        if mod_arg:
            executed.append(mod_arg)
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
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    # setup self-improvement engine stubs
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = module_index_db.ModuleIndexDB(map_path)
    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups.setdefault("1", 1)
        self.save()
    monkeypatch.setattr(module_index_db.ModuleIndexDB, "refresh", fake_refresh)

    generated: list[list[str]] = []
    integrated: list[list[str]] = []
    simulated: list[bool] = []

    def fake_generate(mods):
        generated.append(list(mods))
        return [1]

    def fake_try(mods):
        integrated.append(list(mods))
        return [1]

    def fake_run(*_a, **_k):
        simulated.append(True)

    _integrate_orphans, _refresh_module_map = _load_refresh_methods(
        fake_generate, fake_try, fake_run
    )

    class DummyLogger:
        def exception(self, *a, **k):
            pass
        def info(self, *a, **k):
            pass

    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
        _last_map_refresh=0.0,
        auto_refresh_map=True,
        meta_logger=None,
    )
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)
    engine._test_orphan_modules = types.MethodType(lambda self, paths: set(paths), engine)
    engine._collect_recursive_modules = lambda mods: set(mods)

    # track modules passed to integration callback
    passed: list[str] = []

    def integration(mods: list[str]) -> None:
        passed.extend(mods)
        engine._refresh_module_map(mods)

    svc = self_test_mod.SelfTestService(
        include_orphans=False,
        discover_orphans=False,
        discover_isolated=True,
        recursive_isolated=True,
        integration_callback=integration,
        context_builder=DummyBuilder(),
    )
    asyncio.run(svc._run_once())

    assert called.get("recursive") is True
    assert set(executed) == {"helper.py", "isolated.py"}  # path-ignore
    assert set(passed) == {"helper.py", "isolated.py"}  # path-ignore
    data = json.loads(map_path.read_text()).get("modules", {})
    assert {"helper.py", "isolated.py"}.issubset(data.keys())  # path-ignore
    assert generated and sorted(generated[0]) == ["helper.py", "isolated.py"]  # path-ignore
    assert integrated and sorted(integrated[0]) == ["helper.py", "isolated.py"]  # path-ignore
    assert simulated
    mods_path = tmp_path / "orphan_modules.json"
    if mods_path.exists():
        assert json.loads(mods_path.read_text()) == []


@pytest.mark.skip(reason="Legacy orphan integration path disabled in test environment")
def test_recursive_orphan_integration(monkeypatch, tmp_path):
    # prepare small repo tree
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "helper.py").write_text("VALUE = 2\n")  # path-ignore
    (tmp_path / "orphan.py").write_text("import helper\n")  # path-ignore

    # stub recursive orphan discovery
    called: list[dict[str, object]] = []

    def fake_discover(root, *, module_map=None):
        called.append({"root": root, "module_map": module_map})
        return {"orphan": [], "helper": ["orphan"]}

    import sandbox_runner

    monkeypatch.setattr(sandbox_runner, "discover_recursive_orphans", fake_discover)

    # capture executed paths
    executed: list[str] = []

    async def fake_exec(*cmd, **kwargs):
        mod_arg = None
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.endswith(".py") and mod_arg is None:  # path-ignore
                mod_arg = s
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
        if mod_arg:
            executed.append(mod_arg)
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
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    # setup self-improvement engine stubs
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = module_index_db.ModuleIndexDB(map_path)

    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups.setdefault("1", 1)
        self.save()

    monkeypatch.setattr(module_index_db.ModuleIndexDB, "refresh", fake_refresh)

    generated: list[list[str]] = []
    integrated: list[list[str]] = []
    simulated: list[bool] = []

    def fake_generate(mods):
        generated.append(list(mods))
        return [1]

    def fake_try(mods):
        integrated.append(list(mods))
        return [1]

    def fake_run(*_a, **_k):
        simulated.append(True)

    _integrate_orphans, _refresh_module_map = _load_refresh_methods(
        fake_generate, fake_try, fake_run
    )

    class DummyLogger:
        def exception(self, *a, **k):
            pass

        def info(self, *a, **k):
            pass

    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
        _last_map_refresh=0.0,
        auto_refresh_map=True,
        meta_logger=None,
    )
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)
    engine._test_orphan_modules = types.MethodType(lambda self, paths: set(paths), engine)
    engine._collect_recursive_modules = lambda mods: set(mods)

    passed: list[str] = []

    def integration(mods: list[str]) -> None:
        passed.extend(mods)
        engine._refresh_module_map(mods)

    svc = self_test_mod.SelfTestService(
        include_orphans=False,
        discover_orphans=True,
        recursive_orphans=True,
        clean_orphans=True,
        integration_callback=integration,
        context_builder=DummyBuilder(),
    )
    asyncio.run(svc._run_once())

    assert called and called[0]["root"] == str(tmp_path)
    assert Path(str(called[0]["module_map"])).resolve() == map_path
    assert set(executed) == {"helper.py", "orphan.py"}  # path-ignore
    assert set(passed) == {"helper.py", "orphan.py"}  # path-ignore
    data = json.loads(map_path.read_text()).get("modules", {})
    assert {"helper.py", "orphan.py"}.issubset(data.keys())  # path-ignore
    assert generated and sorted(generated[0]) == ["helper.py", "orphan.py"]  # path-ignore
    assert integrated and sorted(integrated[0]) == ["helper.py", "orphan.py"]  # path-ignore
    assert simulated
    mods_path = data_dir / "orphan_modules.json"
    assert json.loads(mods_path.read_text()) == []


def test_generate_stub_with_scenarios(monkeypatch, tmp_path):
    (tmp_path / "sandbox_data").mkdir()
    module_path = tmp_path / "math_mod.py"  # path-ignore
    module_path.write_text("def add(a, b):\n    return a + b\n")

    scenarios = {
        "fixtures": {"one": 1, "two": 2, "three": 3},
        "tests": {
            "add": [
                {
                    "args": [{"fixture": "one"}, {"fixture": "two"}],
                    "expected": {"fixture": "three"},
                }
            ]
        },
    }

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    svc = self_test_mod.SelfTestService(
        stub_scenarios={str(module_path): scenarios},
        context_builder=DummyBuilder(),
    )
    stub = svc._generate_pytest_stub(str(module_path), scenarios)

    cmd = [sys.executable, "-m", "pytest", "-q", str(stub)]
    proc = subprocess.run(cmd, capture_output=True, cwd=tmp_path)
    output = proc.stdout.decode() + proc.stderr.decode()
    assert proc.returncode == 0, output

    content = stub.read_text()
    assert "SCENARIOS" in content and "FIXTURES" in content

