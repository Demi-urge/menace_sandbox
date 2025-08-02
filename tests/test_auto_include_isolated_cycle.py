import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

from tests.test_recursive_orphans import (
    _load_methods,
    DummyIndex,
    DummyLogger,
)

ROOT = Path(__file__).resolve().parents[1]

_integrate_orphans, _update_orphan_modules, _refresh_module_map = _load_methods()

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"
)
sts = importlib.util.module_from_spec(spec)
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)

def test_auto_include_isolated_cycle(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("import dep\n")
    (repo / "dep.py").write_text("x = 1\n")

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")

    called = {}
    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(path, *, recursive=False):
        called["recursive"] = recursive
        assert Path(path) == repo
        return ["iso.py", "dep.py"]

    mod.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)

    index = DummyIndex(data_dir / "module_map.json")
    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={},
        logger=DummyLogger(),
    )
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)

    _update_orphan_modules(engine)

    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    assert sorted(orphans) == ["dep.py", "iso.py"]
    assert called.get("recursive") is True

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

    svc = sts.SelfTestService(
        include_orphans=True,
        discover_orphans=False,
        discover_isolated=False,
        integration_callback=engine._refresh_module_map,
    )
    svc.run_once()

    data = json.loads((data_dir / "module_map.json").read_text())
    assert set(data.get("modules", {})) == {"iso.py", "dep.py"}
