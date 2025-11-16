import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import json
import types
import time
import sandbox_runner.environment as env


def test_retry_failed_cleanup_success(monkeypatch, tmp_path):
    cid = "c1"
    overlay = tmp_path / "ov"
    overlay.mkdir()
    (overlay / "overlay.qcow2").touch()
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    file.write_text(json.dumps({cid: 0.0, str(overlay): 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)
    monkeypatch.setattr(env.shutil, "rmtree", lambda p: None)

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    env._CLEANUP_RETRY_SUCCESSES = 0
    env._CLEANUP_RETRY_FAILURES = 0
    env._CONSECUTIVE_CLEANUP_FAILURES = 0
    env.retry_failed_cleanup()
    assert json.loads(file.read_text()) == {}
    assert env._CLEANUP_RETRY_SUCCESSES == 2
    assert json.loads(stats_file.read_text())["cleanup_retry_successes"] == 2
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["cleanup_retry_successes_total"] == 2.0
    assert metrics["consecutive_cleanup_failures"] == 0.0


def test_retry_failed_cleanup_failure(monkeypatch, tmp_path):
    cid = "c1"
    overlay = tmp_path / "ov"
    overlay.mkdir()
    (overlay / "overlay.qcow2").touch()
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    file.write_text(json.dumps({cid: 0.0, str(overlay): 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)

    def fail_rmtree(p):
        raise OSError("boom")

    def fail_run(cmd, stdout=None, stderr=None, text=None, check=False):
        return types.SimpleNamespace(returncode=1, stdout="x", stderr="err")

    monkeypatch.setattr(env.shutil, "rmtree", fail_rmtree)
    monkeypatch.setattr(env.subprocess, "run", fail_run)
    monkeypatch.setattr(env, "_rmtree_windows", lambda p, attempts=5, base=0.2: False)
    env._CLEANUP_RETRY_SUCCESSES = 0
    env._CLEANUP_RETRY_FAILURES = 0
    env._CONSECUTIVE_CLEANUP_FAILURES = 0
    env.retry_failed_cleanup()
    remaining = set(json.loads(file.read_text()).keys())
    assert cid in remaining and str(overlay) in remaining
    assert env._CLEANUP_RETRY_FAILURES == 2
    assert json.loads(stats_file.read_text())["cleanup_retry_failures"] == 2
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["cleanup_retry_failures_total"] == 2.0
    assert metrics["consecutive_cleanup_failures"] == 1.0


def test_retry_failed_cleanup_persistent(monkeypatch, tmp_path, caplog):
    cid = "c1"
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    old = time.time() - 120
    file.write_text(json.dumps({cid: old}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)
    monkeypatch.setattr(env, "_FAILED_CLEANUP_ALERT_AGE", 60)

    monkeypatch.setattr(env.shutil, "rmtree", lambda p: (_ for _ in ()).throw(OSError("boom")))

    def fail_run(cmd, stdout=None, stderr=None, text=None, check=False):
        return types.SimpleNamespace(returncode=1, stdout="x", stderr="err")

    monkeypatch.setattr(env.subprocess, "run", fail_run)
    monkeypatch.setattr(env, "_rmtree_windows", lambda p, attempts=5, base=0.2: False)

    logs = []
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: logs.append((issue, success)))
    caplog.set_level("WARNING")

    env.retry_failed_cleanup()

    assert ("cleanup_retry_failure", False) in logs
    assert "persistent cleanup failures" in caplog.text


def test_retry_failed_cleanup_consecutive_alert(monkeypatch, tmp_path, caplog):
    cid = "c1"
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    file.write_text(json.dumps({cid: 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="x"))
    monkeypatch.setattr(env.shutil, "rmtree", lambda p: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(env, "_rmtree_windows", lambda p, attempts=5, base=0.2: False)
    monkeypatch.setattr(env, "_CLEANUP_ALERT_THRESHOLD", 1)
    logs = []
    monkeypatch.setattr(env, "_log_diagnostic", lambda issue, success: logs.append((issue, success)))
    caplog.set_level("ERROR")
    env._CONSECUTIVE_CLEANUP_FAILURES = 0
    env.retry_failed_cleanup()
    env.retry_failed_cleanup()
    assert env._CONSECUTIVE_CLEANUP_FAILURES == 2
    assert ("persistent_cleanup_failure", False) in logs
    assert "cleanup retries failing 2 times consecutively" in caplog.text


def test_retry_failed_cleanup_triggers_prune(monkeypatch, tmp_path, caplog):
    cid = "c1"
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    file.write_text(json.dumps({cid: 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)
    monkeypatch.setattr(env.shutil, "rmtree", lambda p: (_ for _ in ()).throw(OSError("boom")))
    prune_calls = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:3] == ["docker", "system", "prune"]:
            prune_calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=1, stdout="x", stderr="err")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env, "_rmtree_windows", lambda p, attempts=5, base=0.2: False)
    monkeypatch.setattr(env, "_MAX_FAILURE_ATTEMPTS", 0)
    caplog.set_level("WARNING")
    env.retry_failed_cleanup()
    assert prune_calls
    assert "failsafe prune" in caplog.text.lower()
