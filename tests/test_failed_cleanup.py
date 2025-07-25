import json
import types
import time
import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid="x"):
        self.id = cid
    def stop(self, timeout=0):
        pass
    def remove(self, force=True):
        pass


def test_failed_cleanup_record(monkeypatch, tmp_path):
    file = tmp_path / "cleanup.json"
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="x\n"))
    c = DummyContainer("x")
    env._stop_and_remove(c)
    data = json.loads(file.read_text())
    assert list(data.keys()) == ["x"]


def test_report_failed_cleanup(monkeypatch, tmp_path):
    file = tmp_path / "cleanup.json"
    now = time.time() - 120
    file.write_text(json.dumps({"old": now, "new": time.time()}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    logs = {}
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: logs.setdefault("called", True))
    res = env.report_failed_cleanup(threshold=60, alert=True)
    assert "old" in res and "new" not in res
    assert logs.get("called")
