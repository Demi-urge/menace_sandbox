import json
import types
import sandbox_runner.environment as env


def test_recorded_containers_removed(monkeypatch, tmp_path):
    file = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", file)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(file) + ".lock"))
    file.write_text(json.dumps(["a", "b"]))

    removed = []

    def fake_run(cmd, **kw):
        if "ps" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="")
        if "rm" in cmd:
            removed.append(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))

    env.purge_leftovers()

    assert set(removed) == {"a", "b"}
    assert file.exists()
    assert json.loads(file.read_text()) == []


def test_reconcile_removes_untracked(monkeypatch, tmp_path):
    file = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", file)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(file) + ".lock"))
    env._write_active_containers([])

    stray_dir = tmp_path / "stray"
    stray_dir.mkdir()

    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_CREATED.clear()
    env._CONTAINER_DIRS["c123"] = str(stray_dir)

    removed = []

    def fake_run(cmd, **kw):
        if cmd[:4] == ["docker", "ps", "-aq", "--filter"]:
            return types.SimpleNamespace(returncode=0, stdout="c123\n")
        if cmd[:3] == ["docker", "rm", "-f"]:
            removed.append(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env.reconcile_active_containers()

    assert removed == ["c123"]
    assert "c123" not in env._CONTAINER_DIRS
    assert not stray_dir.exists()
    assert json.loads(file.read_text()) == []


def test_record_failure_when_container_persists(monkeypatch, tmp_path):
    file = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", file)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(file) + ".lock"))
    file.write_text(json.dumps(["c1"]))

    recorded = []
    removed_failed = []

    def fake_run(cmd, **kw):
        if cmd[:4] == ["docker", "ps", "-aq", "--filter"] and cmd[4].startswith("id="):
            return types.SimpleNamespace(returncode=0, stdout="c1\n")
        if cmd[:3] == ["docker", "rm", "-f"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "reconcile_active_containers", lambda: None)
    monkeypatch.setattr(env, "_read_active_overlays", lambda: [])
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=False: 0)
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", False)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", False)
    monkeypatch.setattr(env, "_record_failed_cleanup", lambda cid: recorded.append(cid))
    monkeypatch.setattr(env, "_remove_failed_cleanup", lambda cid: removed_failed.append(cid))

    env.purge_leftovers()

    assert recorded == ["c1"]
    assert "c1" not in removed_failed
    assert json.loads(file.read_text()) == ["c1"]
