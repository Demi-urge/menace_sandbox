import types
import importlib
import sys
from pathlib import Path

from tests.test_run_autonomous_env_vars import _load_module


def test_metrics_server_started(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tok")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    called = {}
    monkeypatch.setattr(mod, "start_metrics_server", lambda port: called.setdefault("port", port))

    mod.main(["--check-settings", "--metrics-port", "9999", "--recursive-orphans"])

    assert called.get("port") == 9999
