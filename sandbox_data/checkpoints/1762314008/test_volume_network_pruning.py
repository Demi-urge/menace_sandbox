import types
import sandbox_runner.environment as env


def test_purge_leftovers_prunes_volumes_and_networks(monkeypatch, tmp_path):
    active_containers = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", active_containers)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(active_containers) + ".lock"))
    active_containers.write_text("[]")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    env._write_active_overlays([])
    monkeypatch.setattr(env, "_purge_stale_vms", lambda: 0)

    cmds = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        cmds.append(cmd)
        if cmd[:4] == ["docker", "ps", "-aq", f"--filter"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:3] == ["docker", "volume", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="vol1\n")
        if cmd[:3] == ["docker", "network", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="net1\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", True)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", True)

    env.purge_leftovers()

    assert ["docker", "volume", "ls", "-q", "--filter", f"label={env._POOL_LABEL}=1"] in cmds
    assert ["docker", "volume", "rm", "-f", "vol1"] in cmds
    assert ["docker", "network", "ls", "-q", "--filter", f"label={env._POOL_LABEL}=1"] in cmds
    assert ["docker", "network", "rm", "-f", "net1"] in cmds
