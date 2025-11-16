import types
import sandbox_runner.environment as env


def test_prune_volumes_and_networks(monkeypatch):
    calls = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        calls.append(cmd)
        if cmd[:3] == ["docker", "volume", "ls"]:
            if "--filter" in cmd:
                return types.SimpleNamespace(returncode=0, stdout="vol1\nvol2\n")
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:3] == ["docker", "volume", "rm"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:3] == ["docker", "network", "ls"]:
            if "--filter" in cmd:
                return types.SimpleNamespace(returncode=0, stdout="net1\nnet2\n")
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:3] == ["docker", "network", "rm"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", True)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", True)

    env._CLEANUP_METRICS.clear()

    removed_vols = env._prune_volumes()
    removed_nets = env._prune_networks()

    assert removed_vols == 2
    assert removed_nets == 2
    assert ["docker", "volume", "rm", "-f", "vol1"] in calls
    assert ["docker", "volume", "rm", "-f", "vol2"] in calls
    assert ["docker", "network", "rm", "-f", "net1"] in calls
    assert ["docker", "network", "rm", "-f", "net2"] in calls
    assert env._CLEANUP_METRICS["volume"] == 2
    assert env._CLEANUP_METRICS["network"] == 2
