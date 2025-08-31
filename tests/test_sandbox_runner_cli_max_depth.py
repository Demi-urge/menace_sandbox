import os
import sys
import types

def _load_cli(monkeypatch):
    env_mod = types.SimpleNamespace()
    env_mod.SANDBOX_ENV_PRESETS = [{}]
    env_mod.load_presets = lambda: env_mod.SANDBOX_ENV_PRESETS
    env_mod.simulate_full_environment = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    import importlib
    cli = importlib.reload(__import__("sandbox_runner.cli", fromlist=["dummy"]))
    return cli

def test_cli_sets_max_recursion_depth(monkeypatch):
    cli = _load_cli(monkeypatch)
    monkeypatch.setattr(cli, "_run_sandbox", lambda *a, **k: None)
    monkeypatch.delenv("SANDBOX_MAX_RECURSION_DEPTH", raising=False)
    cli.main(["--max-recursion-depth", "2"])
    assert os.getenv("SANDBOX_MAX_RECURSION_DEPTH") == "2"
