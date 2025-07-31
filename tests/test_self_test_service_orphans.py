import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
)
mod = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
spec.loader.exec_module(mod)


def test_include_orphans(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "orphan_modules.json").write_text(json.dumps(["foo.py", "bar.py"]))

    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    svc = mod.SelfTestService(include_orphans=True)
    asyncio.run(svc._run_once())

    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)
    assert any("bar.py" in c for c in joined)
    assert svc.results.get("orphan_total") == 2
    assert svc.results.get("orphan_failed") == 0


def test_auto_discover_orphans(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types
    helper = types.ModuleType("sandbox_runner")
    helper.discover_orphan_modules = lambda repo, recursive=False: ["foo", "bar"]
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    svc = mod.SelfTestService(include_orphans=True)
    asyncio.run(svc._run_once())

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py", "bar.py"]
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)
    assert any("bar.py" in c for c in joined)


def test_discover_orphans_option(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    helper = types.ModuleType("scripts.find_orphan_modules")
    helper.find_orphan_modules = lambda root, recursive=False: [Path("foo.py"), Path("bar.py")]
    monkeypatch.setitem(sys.modules, "scripts.find_orphan_modules", helper)

    svc = mod.SelfTestService(discover_orphans=True)
    asyncio.run(svc._run_once())

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py", "bar.py"]
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)
    assert any("bar.py" in c for c in joined)


def test_discover_isolated_option(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = lambda root: ["foo.py", "bar.py"]
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)

    svc = mod.SelfTestService(discover_isolated=True)
    asyncio.run(svc._run_once())

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py", "bar.py"]
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)
    assert any("bar.py" in c for c in joined)


def test_recursive_option_used(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    async def fake_exec(*cmd, **kwargs):
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    calls = {}

    def discover(repo):
        calls["used"] = True
        return ["foo"]

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    svc = mod.SelfTestService(include_orphans=True, recursive_orphans=True)
    asyncio.run(svc._run_once(refresh_orphans=True))

    assert calls.get("used") is True


def test_discover_orphans_append(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    async def fake_exec(*cmd, **kwargs):
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    seq = [[Path("foo.py")], [Path("bar.py")]]
    calls = []

    def find(root, recursive=False):
        calls.append(recursive)
        return seq.pop(0)

    helper = types.ModuleType("scripts.find_orphan_modules")
    helper.find_orphan_modules = find
    monkeypatch.setitem(sys.modules, "scripts.find_orphan_modules", helper)

    svc = mod.SelfTestService(discover_orphans=True, recursive_orphans=True)
    asyncio.run(svc._run_once())
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py"]

    asyncio.run(svc._run_once())
    data2 = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data2) == ["bar.py", "foo.py"]
    assert all(c is True for c in calls)


def test_recursive_chain_modules(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("x = 1\n")

    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(" ".join(map(str, cmd)))
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import ast, types, os
    path = ROOT / "sandbox_runner.py"
    src = path.read_text()
    tree = ast.parse(src)
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"discover_recursive_orphans", "discover_orphan_modules", "_discover_orphans_once"}
    ]
    from typing import List, Iterable
    mod_dict = {"ast": ast, "os": os, "json": json, "Path": Path, "List": List, "Iterable": Iterable}
    ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    code = ast.Module(body=funcs, type_ignores=[])
    exec(compile(code, str(path), "exec"), mod_dict)
    discover = mod_dict["discover_recursive_orphans"]

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    svc = mod.SelfTestService(include_orphans=True, recursive_orphans=True)
    asyncio.run(svc._run_once())

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data) == ["a.py", "b.py", "c.py"]
    joined = "\n".join(calls)
    assert "a.py" in joined
    assert "b.py" in joined
    assert "c.py" in joined

