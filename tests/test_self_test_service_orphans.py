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
    assert svc.results.get("orphan_failed") == []


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
    helper = types.ModuleType("scripts.discover_isolated_modules")
    helper.discover_isolated_modules = lambda base=None: ["foo.py", "bar.py"]
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", helper)

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
    helper.find_orphan_modules = lambda root: [Path("foo.py"), Path("bar.py")]
    monkeypatch.setitem(sys.modules, "scripts.find_orphan_modules", helper)

    svc = mod.SelfTestService(discover_orphans=True)
    asyncio.run(svc._run_once())

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py", "bar.py"]
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)
    assert any("bar.py" in c for c in joined)

