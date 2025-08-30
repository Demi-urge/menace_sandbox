import importlib
import json


def test_retry_failed_cleanup_removes_entries(monkeypatch, tmp_path):
    """Environment cleanup retries recorded paths and clears state."""
    env = importlib.reload(importlib.import_module("sandbox_runner.environment"))

    failed_file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    temp_dir = tmp_path / "stale"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("data")

    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", failed_file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)

    env._record_failed_cleanup(str(temp_dir))
    assert failed_file.exists()

    successes, failures = env.retry_failed_cleanup()

    assert successes == 1
    assert failures == 0
    assert not temp_dir.exists()
    assert json.loads(failed_file.read_text()) == {}
