import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

from prometheus_client import REGISTRY

ROOT = Path(__file__).resolve().parents[1]

# Provide minimal stubs for optional heavy dependencies
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

# reset prometheus registry to avoid duplicate metric errors when importing
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


def _setup(monkeypatch, tmp_path, chain_len):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.chdir(tmp_path)

    # create a chain of modules a -> b -> c -> ... depending on chain_len
    prev = None
    for idx in range(chain_len):
        name = chr(ord("a") + idx)
        path = tmp_path / f"{name}.py"  # path-ignore
        if idx < chain_len - 1:
            nxt = chr(ord("a") + idx + 1)
            path.write_text(f"import {nxt}\n")
        else:
            path.write_text("x = 1\n")
        prev = name

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "orphan_modules.json").write_text(json.dumps(["a.py"]))  # path-ignore
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    commands: list[str] = []

    async def fake_exec(*cmd, **kwargs):
        commands.append(" ".join(map(str, cmd)))
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
        (tmp_path / workflows_db).write_text("dummy")
        return [1]

    env_mod = types.ModuleType("environment")
    env_mod.generate_workflows_for_modules = fake_generate
    env_mod.auto_include_modules = lambda mods: fake_generate(mods)
    sr = importlib.import_module("sandbox_runner")
    sr.environment = env_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    import orphan_analyzer

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda p: False)

    def integrate(mods):
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).name] = 1
        map_path.write_text(json.dumps(data))
        from sandbox_runner import environment

        environment.generate_workflows_for_modules([Path(m).name for m in mods])

    return commands, generated, map_path, data_dir, integrate


def _run_and_assert(monkeypatch, tmp_path, chain_len):
    commands, generated, map_path, data_dir, integrate = _setup(monkeypatch, tmp_path, chain_len)

    svc = sts.SelfTestService(
        include_orphans=True,
        integration_callback=integrate,
        clean_orphans=True,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    # All modules should be executed
    cmd = " ".join(commands)
    for idx in range(chain_len):
        name = f"{chr(ord('a') + idx)}.py"  # path-ignore
        assert name in cmd

    # Module map updated
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {f"{chr(ord('a') + i)}.py" for i in range(chain_len)}  # path-ignore

    # Workflows generated for all modules
    assert generated and generated[0] == sorted(data["modules"].keys())
    assert (tmp_path / "workflows.db").exists()

    # Orphan list cleaned
    assert json.loads((data_dir / "orphan_modules.json").read_text()) == []


def test_recursive_self_improvement_simple(tmp_path, monkeypatch):
    _run_and_assert(monkeypatch, tmp_path, 2)


def test_recursive_self_improvement_chain(tmp_path, monkeypatch):
    _run_and_assert(monkeypatch, tmp_path, 3)
