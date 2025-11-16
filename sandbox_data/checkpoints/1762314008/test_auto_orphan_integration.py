import importlib.util
import json
import sys
import types
import asyncio
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
self_test_mod = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
else:
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    sys.modules["menace"] = pkg
from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()
spec.loader.exec_module(self_test_mod)
self_test_mod.analyze_redundancy = lambda p: False

spec_db = importlib.util.spec_from_file_location("module_index_db", ROOT / "module_index_db.py")  # path-ignore
module_index_db = importlib.util.module_from_spec(spec_db)
spec_db.loader.exec_module(module_index_db)
sys.modules["module_index_db"] = module_index_db


def test_default_auto_integration(monkeypatch, tmp_path):
    (tmp_path / "sandbox_data").mkdir()
    (tmp_path / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "isolated.py").write_text("import helper\n")  # path-ignore

    import types
    runner = types.ModuleType("sandbox_runner")

    def fake_discover(repo, module_map=None):
        return {"isolated": [], "helper": ["isolated"]}

    runner.discover_recursive_orphans = fake_discover

    generated: list[list[str]] = []
    integrated: list[list[str]] = []
    simulated: list[bool] = []

    def fake_generate(mods):
        generated.append(list(mods))
        return [1]

    def fake_try(mods):
        integrated.append(list(mods))
        return [1]

    def fake_run():
        simulated.append(True)

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.generate_workflows_for_modules = fake_generate
    env_mod.try_integrate_into_workflows = fake_try
    env_mod.run_workflow_simulations = fake_run
    def fake_auto(mods):
        fake_generate(mods)
        fake_try(mods)
        fake_run()
    env_mod.auto_include_modules = fake_auto

    runner.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    map_path = tmp_path / "sandbox_data" / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups.setdefault("1", 1)
        self.save()

    monkeypatch.setattr(module_index_db.ModuleIndexDB, "refresh", fake_refresh)

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
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
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path / "sandbox_data"))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            if k.get("return_metadata"):
                return "", {}
            return ""

    svc = self_test_mod.SelfTestService(context_builder=DummyBuilder())
    svc.run_once()

    data = json.loads(map_path.read_text())
    assert set(data["modules"].keys()) == {"isolated.py", "helper.py"}  # path-ignore
    assert generated and generated[0] == ["helper.py", "isolated.py"]  # path-ignore
    assert integrated and integrated[0] == ["helper.py", "isolated.py"]  # path-ignore
    assert simulated and simulated[0] is True
