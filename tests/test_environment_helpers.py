import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
if "sqlalchemy" not in sys.modules:
    sa = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_mod.Engine = object
    sa.engine = engine_mod
    sys.modules["sqlalchemy"] = sa
    sys.modules["sqlalchemy.engine"] = engine_mod
import sandbox_runner.environment as env


def test_parse_failure_modes():
    assert env._parse_failure_modes("disk,network") == {"disk", "network"}
    assert env._parse_failure_modes(["cpu_spike", "memory"]) == {"cpu_spike", "memory"}


def test_inject_failure_modes_disk():
    snippet = "open('f','w').write('x')"
    out = env._inject_failure_modes(snippet, {"disk"})
    assert "_orig_open" in out
    assert "open('f','w').write('x')" in out


def test_generate_input_stubs_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_INPUT_STUBS", '[{"a": 1}]')
    import importlib
    importlib.reload(env)
    stubs = env.generate_input_stubs()
    assert stubs == [{"a": 1}]


def test_generate_input_stubs_random(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "random")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib
    importlib.reload(env)
    monkeypatch.setattr(env.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(env.random, "randint", lambda a, b: a)
    monkeypatch.setattr(env.random, "random", lambda: 0.0)
    stubs = env.generate_input_stubs(2)
    assert len(stubs) == 2
    for stub in stubs:
        assert stub.get("mode") == "default" and stub.get("level") == 1
        assert "flag" in stub


def test_generate_input_stubs_templates(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    f = tmp_path / "temps.json"
    f.write_text('[{"mode": "x", "level": 9}]')
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "templates")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", str(f))
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib
    importlib.reload(env)
    stubs = env.generate_input_stubs(1)
    assert stubs == [{"mode": "x", "level": 9}]

