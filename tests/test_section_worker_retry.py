import os
import asyncio
import types
import sys
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
dummy_jinja = types.ModuleType("jinja2")
dummy_jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", dummy_jinja)
dummy_yaml = types.ModuleType("yaml")
dummy_yaml.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", dummy_yaml)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
import sandbox_runner.environment as env


def test_section_worker_retry(monkeypatch):
    monkeypatch.setattr(env, "_rlimits_supported", lambda: False)
    monkeypatch.setattr(env, "psutil", None)

    calls = []

    async def fake_exec(code, env_input, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("fail")
        return {"exit_code": 0.0, "cpu": 0.0, "memory": 0.0, "disk_io": 0.0}

    monkeypatch.setattr(env, "_execute_in_container", fake_exec)
    monkeypatch.setattr(env.time, "sleep", lambda s: None)

    res, _ = asyncio.run(env._section_worker("print('x')", {}, 0.0))
    assert len(calls) >= 2
    assert res["exit_code"] == 0.0
