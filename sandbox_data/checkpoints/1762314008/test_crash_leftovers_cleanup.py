import json
import types
import sandbox_runner.environment as env


def test_crash_leftovers_cleanup(monkeypatch, tmp_path):
    # patch active files
    containers = tmp_path / "active.json"
    overlays = tmp_path / "overlays.json"
    failed = tmp_path / "failed.json"
    stats = tmp_path / "stats.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", containers)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(containers) + ".lock"))
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))
    monkeypatch.setattr(env, "_FAILED_OVERLAYS_FILE", failed)
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", tmp_path / "failed_cleanup.json")
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats)

    # record stale resources
    containers.write_text(json.dumps(["c1", "c2"]))
    overlay_rec = tmp_path / "ov_rec"
    overlay_rec.mkdir()
    (overlay_rec / "overlay.qcow2").touch()
    env._write_active_overlays([str(overlay_rec)])
    overlay_left = tmp_path / "ov_left"
    overlay_left.mkdir()
    (overlay_left / "overlay.qcow2").touch()

    # configure environment
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", True)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", True)
    env._STALE_CONTAINERS_REMOVED = 0
    env._STALE_VMS_REMOVED = 0

    cmds = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        cmds.append(cmd)
        if cmd[:4] == ["docker", "ps", "-aq", "--filter"]:
            if cmd[4].startswith("id="):
                return types.SimpleNamespace(returncode=0, stdout="")
            return types.SimpleNamespace(returncode=0, stdout="c3\n")
        if cmd[:3] == ["docker", "volume", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="vol1\n")
        if cmd[:3] == ["docker", "network", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="net1\n")
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env.purge_leftovers()

    # stale containers removed and file cleaned
    assert json.loads(containers.read_text()) == []
    assert ["docker", "rm", "-f", "c1"] in cmds
    assert ["docker", "rm", "-f", "c2"] in cmds
    assert ["docker", "rm", "-f", "c3"] in cmds

    # overlay directories cleaned
    assert not overlay_rec.exists()
    assert not overlay_left.exists()
    assert json.loads(overlays.read_text()) == []

    # volume and network pruning commands issued
    assert ["docker", "volume", "ls", "-q", "--filter", f"label={env._POOL_LABEL}=1"] in cmds
    assert ["docker", "volume", "rm", "-f", "vol1"] in cmds
    assert ["docker", "network", "ls", "-q", "--filter", f"label={env._POOL_LABEL}=1"] in cmds
    assert ["docker", "network", "rm", "-f", "net1"] in cmds

    # metrics updated
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["stale_containers_removed"] == 3.0
    assert metrics["stale_vms_removed"] >= 1.0
