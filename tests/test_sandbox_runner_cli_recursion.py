import os
import sys
import types


def _load_cli(monkeypatch):
    env_mod = types.SimpleNamespace()
    env_mod.SANDBOX_ENV_PRESETS = [{}]
    env_mod.load_presets = lambda: env_mod.SANDBOX_ENV_PRESETS
    env_mod.simulate_full_environment = lambda *a, **k: None
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
        discover_isolated = True
        env_di = os.getenv("SANDBOX_DISCOVER_ISOLATED")
        if env_di is not None:
            discover_isolated = env_di.lower() in {"1", "true"}
        arg_di = getattr(args, "discover_isolated", None)
        if arg_di is not None:
            discover_isolated = arg_di
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = (
            "1" if discover_isolated else "0"
        )
        recursive_isolated = True
        env_iso = os.getenv("SANDBOX_RECURSIVE_ISOLATED")
        if env_iso is not None and env_iso.lower() in {"0", "false", "no"}:
            recursive_isolated = False
        arg_iso = getattr(args, "recursive_isolated", None)
        if arg_iso is not None:
            recursive_isolated = arg_iso
        os.environ["SANDBOX_RECURSIVE_ISOLATED"] = (
            "1" if recursive_isolated else "0"
        )
        capture["recursive_isolated"] = recursive_isolated
        capture["discover_isolated"] = discover_isolated

    monkeypatch.setattr(cli, "_run_sandbox", fake_run)


def test_cli_recursion_default(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ORPHANS", raising=False)
    monkeypatch.delenv("SELF_TEST_RECURSIVE_ORPHANS", raising=False)
    cli.main([])
    assert capture.get("recursive_orphans") is True
    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "1"


def test_cli_recursion_flag_overrides_env(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "1")
    monkeypatch.setenv("SELF_TEST_RECURSIVE_ORPHANS", "1")
    cli.main(["--no-recursive-include"])
    assert capture.get("recursive_orphans") is False
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "0"
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")
    monkeypatch.setenv("SELF_TEST_RECURSIVE_ORPHANS", "0")
    cli.main(["--recursive-include"])
    assert capture.get("recursive_orphans") is True
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "1"


def test_cli_recursive_orphans_alias(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.setenv("SANDBOX_RECURSIVE_ORPHANS", "0")
    monkeypatch.setenv("SELF_TEST_RECURSIVE_ORPHANS", "0")
    cli.main(["--recursive-orphans"])
    assert capture.get("recursive_orphans") is True
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "1"


def test_cli_recursive_isolated_sets_env(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.delenv("SELF_TEST_RECURSIVE_ISOLATED", raising=False)
    cli.main(["--recursive-isolated"])
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"


def test_cli_auto_include_isolated_enables_recursion(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.delenv("SANDBOX_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_DISCOVER_ISOLATED", raising=False)
    monkeypatch.delenv("SELF_TEST_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SELF_TEST_RECURSIVE_ISOLATED", raising=False)
    cli.main(["--auto-include-isolated"])
    assert os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED") == "1"
    assert capture.get("discover_isolated") is True
    assert capture.get("recursive_isolated") is True
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "1"


def test_cli_no_discover_isolated_disables(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    cli.main(["--no-discover-isolated"])
    assert capture.get("discover_isolated") is False
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "0"


def test_cli_discover_isolated_overrides_env(monkeypatch):
    capture = {}
    cli = _load_cli(monkeypatch)
    _capture_run(monkeypatch, cli, capture)
    monkeypatch.setenv("SANDBOX_DISCOVER_ISOLATED", "0")
    cli.main(["--discover-isolated"])
    assert capture.get("discover_isolated") is True
    assert os.getenv("SANDBOX_DISCOVER_ISOLATED") == "1"
