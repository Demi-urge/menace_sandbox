import os
import sys
import types


def _load_cli(monkeypatch):
    env_mod = types.SimpleNamespace(
        SANDBOX_ENV_PRESETS=[{}],
        simulate_full_environment=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    import sandbox_runner.cli as cli

    return cli


def _capture_run(monkeypatch, cli, capture):
    def fake_run(args):
        recursive_orphans = True
        env_rec = os.getenv("SANDBOX_RECURSIVE_ORPHANS")
        if env_rec is not None:
            recursive_orphans = env_rec.lower() in {"1", "true", "yes"}
        arg_rec = getattr(args, "recursive_orphans", None)
        if arg_rec is not None:
            recursive_orphans = arg_rec
        os.environ["SANDBOX_RECURSIVE_ORPHANS"] = "1" if recursive_orphans else "0"
        capture["recursive_orphans"] = recursive_orphans
    monkeypatch.setattr(cli, "_run_sandbox", fake_run)


def test_cli_recursion_default(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ORPHANS", raising=False)
    cli.main([])
    assert capture.get("recursive_orphans") is True


def test_cli_recursion_flag_overrides_env(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "1")
    cli.main(["--no-recursive-orphans"])
    assert capture.get("recursive_orphans") is False
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")
    cli.main(["--recursive-orphans"])
    assert capture.get("recursive_orphans") is True
