import ast
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import types

ROOT = Path(__file__).resolve().parents[1]

# Load SelfTestService dynamically
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",  # path-ignore
)
svc_mod = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
me = types.SimpleNamespace(Gauge=lambda *a, **k: object())
sys.modules["metrics_exporter"] = me
sys.modules["menace.metrics_exporter"] = me
db_mod = types.ModuleType("data_bot")
db_mod.DataBot = object
sys.modules["data_bot"] = db_mod
sys.modules["menace.data_bot"] = db_mod
err_db_mod = types.ModuleType("error_bot")
class _ErrDB:
    def __init__(self, *a, **k):
        pass
err_db_mod.ErrorDB = _ErrDB
sys.modules["error_bot"] = err_db_mod
sys.modules["menace.error_bot"] = err_db_mod
err_log_mod = types.ModuleType("error_logger")
class _ErrLogger:
    def __init__(self, *a, **k):
        pass
err_log_mod.ErrorLogger = _ErrLogger
sys.modules["error_logger"] = err_log_mod
sys.modules["menace.error_logger"] = err_log_mod
ae_mod = types.ModuleType("auto_env_setup")
ae_mod.get_recursive_isolated = lambda: True
ae_mod.set_recursive_isolated = lambda val: None
sys.modules["auto_env_setup"] = ae_mod
sys.modules["menace.auto_env_setup"] = ae_mod
spec.loader.exec_module(svc_mod)
svc_mod.analyze_redundancy = lambda p: False


# simple context builder stub for tests
class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        if k.get("return_metadata"):
            return "", {}
        return ""

# ----------------------------------------------------------------------
# Helper to compile discover_isolated_modules with stubs

def _load_discover(find_func, import_func, recursive_func):
    path = ROOT / "scripts" / "discover_isolated_modules.py"  # path-ignore
    src = path.read_text()
    tree = ast.parse(src)
    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "discover_isolated_modules":
            func = node
            break
    assert func is not None
    mod = {
        "os": __import__("os"),
        "Path": Path,
        "find_orphan_modules": find_func,
        "_discover_import_orphans": import_func,
        "_discover_recursive_orphans": recursive_func,
        "List": __import__("typing").List,
        "analyze_redundancy": lambda p: False,
    }
    ast.fix_missing_locations(func)
    code = ast.Module(body=[func], type_ignores=[])
    exec(compile(code, str(path), "exec"), mod)
    return mod["discover_isolated_modules"]


# ----------------------------------------------------------------------

def test_recursive_import_includes_dependencies(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore

    calls = {"rec": 0, "imp": 0}

    def find_orphans(root, recursive=False):
        assert Path(root) == tmp_path
        return [Path("a.py")]  # path-ignore

    def import_orphans(root, recursive=False):
        calls["imp"] += 1
        assert recursive is False
        return ["a"]

    def recursive_orphans(root):
        calls["rec"] += 1
        return {"a": [], "b": ["a"]}

    discover = _load_discover(find_orphans, import_orphans, recursive_orphans)
    res = discover(str(tmp_path), recursive=True)
    assert calls["rec"] == 1
    assert calls["imp"] == 1
    assert sorted(res) == ["a.py", "b.py"]  # path-ignore


# ----------------------------------------------------------------------

async def _fake_proc(*cmd, **kwargs):
    path = None
    for i, a in enumerate(cmd):
        s = str(a)
        if s.startswith("--json-report-file"):
            path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
            break
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"summary": {"passed": 0, "failed": 0}}, fh)

    class P:
        returncode = 0

        async def communicate(self):
            return b"", b""

        async def wait(self):
            return None

    return P()


def _setup_isolated(monkeypatch, returned):
    import types

    called = {}

    def discover(root, *, recursive=True):
        called["recursive"] = recursive
        called["root"] = root
        return list(returned)

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_proc)
    return called


def test_service_recursive_isolated_updates_file(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)

    called = _setup_isolated(monkeypatch, ["foo.py", "bar.py"])  # path-ignore
    svc = svc_mod.SelfTestService(discover_isolated=True, recursive_isolated=True, context_builder=DummyBuilder())
    asyncio.run(svc._run_once())

    assert called.get("recursive") is True
    assert Path(called.get("root")) == tmp_path
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data) == ["bar.py", "foo.py"]  # path-ignore


def test_settings_enable_isolated_processing(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)

    called = _setup_isolated(monkeypatch, ["one.py"])  # path-ignore
    monkeypatch.setenv("SELF_TEST_DISCOVER_ORPHANS", "0")
    monkeypatch.setenv("SELF_TEST_RECURSIVE_ORPHANS", "0")
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")

    class DummySettings:
        auto_include_isolated = True
        recursive_isolated = True

    monkeypatch.setattr(svc_mod, "SandboxSettings", lambda: DummySettings())
    svc = svc_mod.SelfTestService(discover_isolated=False, recursive_isolated=False, context_builder=DummyBuilder())
    asyncio.run(svc._run_once())

    assert called.get("recursive") is True
    assert Path(called.get("root")) == tmp_path


def test_discover_isolated_records_dependencies(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("x = 1\n")  # path-ignore

    import types

    def fake_discover(root, recursive=True):
        assert recursive is True
        return [Path("a.py")]  # path-ignore

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = fake_discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    svc = svc_mod.SelfTestService(
        discover_orphans=False,
        discover_isolated=True,
        recursive_isolated=True,
        context_builder=DummyBuilder(),
    )
    mods = svc._discover_isolated()
    assert set(mods) == {"a.py", "b.py"}  # path-ignore
    assert svc.orphan_traces.get("b.py", {}).get("parents") == ["a.py"]  # path-ignore


def test_discover_isolated_skips_redundant(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("# deprecated\n")  # path-ignore

    import types

    def fake_discover(root, recursive=True):
        assert recursive is True
        return [Path("a.py")]  # path-ignore

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = fake_discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)

    def fake_analyze(path: Path) -> bool:
        return path.name == "b.py"  # path-ignore

    monkeypatch.setattr(svc_mod, "analyze_redundancy", fake_analyze)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    svc = svc_mod.SelfTestService(
        discover_orphans=False,
        discover_isolated=True,
        recursive_isolated=True,
        context_builder=DummyBuilder(),
    )
    svc.logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    mods = svc._discover_isolated()
    assert mods == ["a.py"]  # path-ignore
    assert svc.orphan_traces["b.py"]["parents"] == ["a.py"]  # path-ignore
    assert svc.orphan_traces["b.py"]["redundant"] is True  # path-ignore
