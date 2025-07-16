import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
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

