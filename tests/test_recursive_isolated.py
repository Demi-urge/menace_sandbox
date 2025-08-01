import ast
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Load SelfTestService dynamically
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
)
svc_mod = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
spec.loader.exec_module(svc_mod)


# ----------------------------------------------------------------------
# Helper to compile discover_isolated_modules with stubs

def _load_discover(find_func, import_func, recursive_func):
    path = ROOT / "scripts" / "discover_isolated_modules.py"
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
    }
    ast.fix_missing_locations(func)
    code = ast.Module(body=[func], type_ignores=[])
    exec(compile(code, str(path), "exec"), mod)
    return mod["discover_isolated_modules"]


# ----------------------------------------------------------------------

def test_recursive_import_includes_dependencies(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    calls = {"rec": 0, "imp": 0}

    def find_orphans(root, recursive=False):
        assert Path(root) == tmp_path
        return [Path("a.py")]

    def import_orphans(root, recursive=False):
        calls["imp"] += 1
        return ["a"]

    def recursive_orphans(root):
        calls["rec"] += 1
        return ["a", "b"]

    discover = _load_discover(find_orphans, import_orphans, recursive_orphans)
    res = discover(str(tmp_path), recursive=True)
    assert calls["rec"] == 1
    assert calls["imp"] == 0
    assert sorted(res) == ["a.py", "b.py"]


# ----------------------------------------------------------------------

async def _fake_proc(*cmd, **kwargs):
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

    def discover(root, *, recursive=False):
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

    called = _setup_isolated(monkeypatch, ["foo.py", "bar.py"])
    svc = svc_mod.SelfTestService(discover_isolated=True, recursive_isolated=True)
    asyncio.run(svc._run_once())

    assert called.get("recursive") is True
    assert Path(called.get("root")) == tmp_path
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data) == ["bar.py", "foo.py"]
