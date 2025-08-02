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
        if env_rec is not None and env_rec.lower() in {"0", "false"}:
            recursive_orphans = False
        arg_rec = getattr(args, "recursive_orphans", None)
        if arg_rec is not None:
            recursive_orphans = arg_rec
        os.environ["SANDBOX_RECURSIVE_ORPHANS"] = "1" if recursive_orphans else "0"
        capture["recursive_orphans"] = recursive_orphans

        if os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED") == "1":
            os.environ.setdefault("SANDBOX_DISCOVER_ISOLATED", "1")
            os.environ.setdefault("SANDBOX_RECURSIVE_ISOLATED", "1")
        recursive_isolated = False
        env_iso = os.getenv("SANDBOX_RECURSIVE_ISOLATED")
        if env_iso is not None and env_iso.lower() in {"1", "true"}:
            recursive_isolated = True
        arg_iso = getattr(args, "recursive_isolated", None)
        if arg_iso is not None:
            recursive_isolated = arg_iso
        os.environ["SANDBOX_RECURSIVE_ISOLATED"] = (
            "1" if recursive_isolated else "0"
        )
        capture["recursive_isolated"] = recursive_isolated
        capture["discover_isolated"] = os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"

    monkeypatch.setattr(cli, "_run_sandbox", fake_run)


def test_cli_recursion_default(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ORPHANS", raising=False)
    cli.main([])
    assert capture.get("recursive_orphans") is True
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "1"


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


def test_cli_recursive_isolated_sets_env(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    cli.main(["--recursive-isolated"])
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"


def test_cli_auto_include_isolated_enables_recursion(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_DISCOVER_ISOLATED", raising=False)
    cli.main(["--auto-include-isolated"])
    assert os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED") == "1"
    assert capture.get("discover_isolated") is True
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
