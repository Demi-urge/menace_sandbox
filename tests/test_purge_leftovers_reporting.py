import json
import sys
import types
import importlib.util
import importlib.machinery
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("sandbox_runner")
pkg.__path__ = [str(ROOT / "sandbox_runner")]
pkg.__spec__ = importlib.machinery.ModuleSpec("sandbox_runner", loader=None, is_package=True)
sys.modules["sandbox_runner"] = pkg

_ENV_PATH = ROOT / "sandbox_runner" / "environment.py"
_SPEC = importlib.util.spec_from_file_location("sandbox_runner.environment", _ENV_PATH)
env = importlib.util.module_from_spec(_SPEC)
sys.modules["sandbox_runner.environment"] = env
assert _SPEC.loader is not None
_SPEC.loader.exec_module(env)

class DummyLock:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc, tb):
        pass

def test_purge_leftovers_reports_failed_cleanup(monkeypatch, tmp_path, caplog):
    file = tmp_path / "failed.json"
    file.write_text(json.dumps({"c1": 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_PURGE_FILE_LOCK", DummyLock())
    monkeypatch.setattr(env, "_read_active_containers", lambda: [])
    monkeypatch.setattr(env, "_read_active_overlays", lambda: [])
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=False: 0)
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", False)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", False)
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    logs = []
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: logs.append((issue, success)))
    caplog.set_level("ERROR")
    env.purge_leftovers()
    assert ("failed_cleanup", False) in logs
    assert "failed cleanup items" in caplog.text
