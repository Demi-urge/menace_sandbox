import asyncio
import importlib.util
import json
import sys
from pathlib import Path
import pytest

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
    mod_find = types.ModuleType("scripts.find_orphan_modules")
    mod_find.find_orphan_modules = lambda root: [Path("foo.py"), Path("bar.py")]
    monkeypatch.setitem(sys.modules, "scripts.find_orphan_modules", mod_find)

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

    def discover(repo, module_map=None):
        calls["used"] = True
        calls["repo"] = repo
        calls["map"] = module_map
        return ["foo"]

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", types.ModuleType("env"))

    svc = mod.SelfTestService(include_orphans=True, recursive_orphans=True)
    asyncio.run(svc._run_once(refresh_orphans=True))

    assert calls.get("used") is True
    assert Path(calls["repo"]) == tmp_path
    assert Path(calls["map"]).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


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

    seq = [["foo"], ["bar"]]
    calls = []

    def discover(repo, module_map=None):
        calls.append((repo, module_map))
        return seq.pop(0)

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", types.ModuleType("env"))

    svc = mod.SelfTestService(discover_orphans=True, recursive_orphans=True)
    asyncio.run(svc._run_once())
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py"]

    asyncio.run(svc._run_once())
    data2 = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data2) == ["bar.py", "foo.py"]
    assert len(calls) == 2
    for repo, m in calls:
        assert Path(repo) == tmp_path
        assert Path(m).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


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

    import types

    params = {}

    def discover(repo, module_map=None):
        params.setdefault("repo", repo)
        params.setdefault("map", module_map)
        return ["a", "b", "c"]

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
    assert Path(params["repo"]) == tmp_path
    assert Path(params["map"]).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


def test_recursive_orphan_multi_scan(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    calls = []
    discover_calls = []

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

    import types

    def discover(repo, module_map=None):
        discover_calls.append((repo, module_map))
        return ["a", "b"]

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    svc = mod.SelfTestService(include_orphans=True, recursive_orphans=True)
    asyncio.run(svc._run_once(refresh_orphans=True))

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data) == ["a.py", "b.py"]
    joined = "\n".join(calls)
    assert "a.py" in joined
    assert "b.py" in joined
    assert len(discover_calls) == 1
    repo, m = discover_calls[0]
    assert Path(repo) == tmp_path
    assert Path(m).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


@pytest.mark.parametrize("var", ["SELF_TEST_INCLUDE_ORPHANS", "SANDBOX_INCLUDE_ORPHANS"])
def test_env_orphans_enabled(tmp_path, monkeypatch, var):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(var, "1")

    svc = mod.SelfTestService()

    assert svc.include_orphans is True

