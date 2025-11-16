import types
import sandbox_runner.environment as env


def test_volume_network_cleanup(monkeypatch, tmp_path):
    volumes = {"vol1"}
    networks = {"net1"}

    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", True)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", True)
    monkeypatch.setattr(env, "psutil", None)

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:3] == ["docker", "volume", "ls"]:
            if "--filter" in cmd:
                return types.SimpleNamespace(returncode=0, stdout="vol1\n")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["docker", "volume", "rm"]:
            volumes.discard(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["docker", "network", "ls"]:
            if "--filter" in cmd:
                return types.SimpleNamespace(returncode=0, stdout="net1\n")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["docker", "network", "rm"]:
            networks.discard(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env.purge_leftovers()

    assert not volumes
    assert not networks
