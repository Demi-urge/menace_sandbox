import json
import types
import sandbox_runner.environment as env

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
