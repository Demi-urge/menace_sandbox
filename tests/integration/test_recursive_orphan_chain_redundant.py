import importlib
import importlib.util
import asyncio
import json
import sys
import types
from dataclasses import dataclass as _dc
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Provide minimal stubs for optional dependencies

pyd = types.ModuleType("pydantic")
sub = types.ModuleType("pydantic.dataclasses")
sub.dataclass = _dc
pyd.BaseModel = type("BaseModel", (), {})
pyd.dataclasses = sub
sys.modules.setdefault("pydantic", pyd)
sys.modules.setdefault("pydantic.dataclasses", sub)

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: object()
sys.modules.setdefault("jinja2", jinja_mod)

yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)

from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()

# load SelfTestService from source
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
)
sts = importlib.util.module_from_spec(spec)
sys.modules.setdefault("menace", types.ModuleType("menace"))
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}


# ---------------------------------------------------------------------------

def test_orphan_chain_skips_redundant(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    # create orphan chain with deprecated leaf: a -> b -> c(deprecated)
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("# deprecated\nVALUE = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    monkeypatch.chdir(tmp_path)

    sr = importlib.import_module("sandbox_runner")
    mapping = sr.discover_recursive_orphans(str(tmp_path), module_map=str(map_path))
    assert mapping == {
        "a": {"parents": [], "redundant": False},
        "b": {"parents": ["a"], "redundant": False},
        "c": {"parents": ["b"], "redundant": True},
    }

    generated: list[list[str]] = []

    def fake_generate(mods, workflows_db="workflows.db"):
        generated.append(list(mods))
        return [1]

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.generate_workflows_for_modules = fake_generate
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

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
    monkeypatch.setenv("SANDBOX_CLEAN_ORPHANS", "1")

    def integrate(mods: list[str]) -> None:
        data = json.loads(map_path.read_text())
        for m in mods:
            name = Path(m).name
            data["modules"][name] = 1
            env_mod.generate_workflows_for_modules([name])
        map_path.write_text(json.dumps(data))

    svc = sts.SelfTestService(
        include_orphans=True,
        recursive_orphans=True,
        clean_orphans=True,
        integration_callback=integrate,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    assert [sorted(g) for g in generated] == [["a.py"], ["b.py"]]  # path-ignore

    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"a.py", "b.py"}  # path-ignore
    orphan_list = json.loads((data_dir / "orphan_modules.json").read_text())
    assert orphan_list == []
