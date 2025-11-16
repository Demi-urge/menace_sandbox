import importlib
import importlib.util
from pathlib import Path
import json
import subprocess
import sys
import types
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("dynamic_path_router", ROOT / "dynamic_path_router.py")
dynamic_path_router = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(dynamic_path_router)  # type: ignore[union-attr]
sys.modules["dynamic_path_router"] = dynamic_path_router

env = importlib.import_module("menace_sandbox.sandbox_runner.environment")
sys.modules["sandbox_runner.environment"] = env

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
    assert isinstance(data["x"], dict)
    assert data["x"].get("reason") == ""


def test_report_failed_cleanup(monkeypatch, tmp_path, caplog):
    file = tmp_path / "cleanup.json"
    now = time.time() - 120
    file.write_text(
        json.dumps(
            {
                "old": {"ts": now, "reason": "stuck"},
                "new": {"ts": time.time(), "reason": "fresh"},
            }
        )
    )
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    logs = {}
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: logs.setdefault("called", True))
    caplog.set_level("ERROR")
    res = env.report_failed_cleanup(threshold=60, alert=True)
    assert "old" in res and "new" not in res
    assert res["old"]["reason"] == "stuck"
    assert logs.get("called")
    assert "failed cleanup items" in caplog.text


def test_retry_records_timeout_reason(monkeypatch, tmp_path, caplog):
    file = tmp_path / "cleanup.json"
    entry = {"ts": time.time() - 120, "reason": ""}
    file.write_text(json.dumps({"container:abc": entry}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)

    def fake_run(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    caplog.set_level("WARNING")

    successes, failures = env.retry_failed_cleanup()

    assert successes == 0 and failures == 1
    data = json.loads(file.read_text())
    assert "timeout" in data["container:abc"]["reason"].lower()
    assert "timed out" in caplog.text
