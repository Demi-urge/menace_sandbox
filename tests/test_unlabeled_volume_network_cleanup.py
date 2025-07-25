import json
import types
import sandbox_runner.environment as env


def test_unlabeled_volume_network_cleanup(monkeypatch, tmp_path):
    active_containers = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", active_containers)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(active_containers) + ".lock"))
    active_containers.write_text("[]")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    env._write_active_overlays([])
    monkeypatch.setattr(env, "_purge_stale_vms", lambda: 0)

    monkeypatch.setattr(env, "_PRUNE_VOLUMES", True)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", True)
    monkeypatch.setattr(env, "_CONTAINER_MAX_LIFETIME", 1.0)
    monkeypatch.setattr(env.time, "time", lambda: 100.0)

    cmds = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        cmds.append(cmd)
        if cmd[:3] == ["docker", "volume", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="vol1\n")
        if cmd[:3] == ["docker", "network", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="net1\n")
        if cmd[:3] == ["docker", "volume", "inspect"]:
            data = [{"CreatedAt": "1970-01-01T00:00:00Z", "Labels": None}]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(data))
        if cmd[:3] == ["docker", "network", "inspect"]:
            data = [{"Created": "1970-01-01T00:00:00Z", "Labels": None, "Name": "net1"}]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(data))
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    env._CLEANUP_METRICS.clear()

    env.purge_leftovers()

    assert ["docker", "volume", "rm", "-f", "vol1"] in cmds
    assert ["docker", "network", "rm", "-f", "net1"] in cmds
    assert env._CLEANUP_METRICS["volume"] == 1
    assert env._CLEANUP_METRICS["network"] == 1
