import types
import importlib
import sys
import os
from pathlib import Path

from tests.test_run_autonomous_env_vars import _load_module


def test_metrics_server_started(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    called = {}
    monkeypatch.setattr(mod, "start_metrics_server", lambda port: called.setdefault("port", port))

    mod.main(["--check-settings", "--metrics-port", "9999", "--no-recursive-include", "--no-recursive-isolated"])

    assert called.get("port") == 9999


def test_recursive_flags_mirrored(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    for key in [
        "SANDBOX_RECURSIVE_ORPHANS",
        "SELF_TEST_RECURSIVE_ORPHANS",
        "SANDBOX_RECURSIVE_ISOLATED",
        "SELF_TEST_RECURSIVE_ISOLATED",
    ]:
        monkeypatch.delenv(key, raising=False)

    mod.main(["--check-settings", "--no-recursive-include", "--no-recursive-isolated"])

    assert os.getenv("SANDBOX_RECURSIVE_ORPHANS") == "0"
    assert os.getenv("SELF_TEST_RECURSIVE_ORPHANS") == "0"
    assert os.getenv("SANDBOX_RECURSIVE_ISOLATED") == "0"
    assert os.getenv("SELF_TEST_RECURSIVE_ISOLATED") == "0"


def test_auto_include_flag_mirrored(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    for key in [
        "SANDBOX_AUTO_INCLUDE_ISOLATED",
        "SELF_TEST_AUTO_INCLUDE_ISOLATED",
    ]:
        monkeypatch.delenv(key, raising=False)

    mod.main(["--check-settings", "--auto-include-isolated"])

    assert os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED") == "1"
    assert os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED") == "1"
