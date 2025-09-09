import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# provide minimal stubs for optional dependencies
pyd_mod = types.ModuleType("pydantic")
sub = types.ModuleType("pydantic.dataclasses")
from dataclasses import dataclass as _dc
sub.dataclass = _dc
pyd_mod.BaseModel = type("BaseModel", (), {})
pyd_mod.dataclasses = sub
sys.modules.setdefault("pydantic", pyd_mod)
sys.modules.setdefault("pydantic.dataclasses", sub)

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: object()
sys.modules.setdefault("jinja2", jinja_mod)

yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)

# reset prometheus registry to avoid duplicate metrics
from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()

# load SelfTestService from source
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
sts = importlib.util.module_from_spec(spec)
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}


def test_recursive_orphan_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    # create dummy modules: a -> b -> c
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("x = 1\n")  # path-ignore

    # verify full dependency chain is discovered
    sr = importlib.import_module("sandbox_runner")
    assert sr.discover_recursive_orphans(str(tmp_path)) == {
        "a": {"parents": [], "redundant": False},
        "b": {"parents": ["a"], "redundant": False},
        "c": {"parents": ["b"], "redundant": False},
    }

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "orphan_modules.json").write_text(json.dumps(["a.py"]))  # path-ignore
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            Path(path).write_text(json.dumps({"summary": {"passed": 1, "failed": 0}}))

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    generated: list[list[str]] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(sorted(mods))
        return [1]

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = fake_generate
    sr.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    def integrate(mods: list[str]) -> None:
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).name] = 1
        map_path.write_text(json.dumps(data))
        from sandbox_runner import environment

        environment.generate_workflows_for_modules([Path(m).name for m in mods])

    monkeypatch.chdir(tmp_path)
    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=integrate,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    assert generated and generated[0] == ["a.py", "b.py", "c.py"]  # path-ignore
    data = json.loads(map_path.read_text())
    assert all(name in data["modules"] for name in ["a.py", "b.py", "c.py"])  # path-ignore
