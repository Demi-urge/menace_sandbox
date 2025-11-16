import json
import types
import sandbox_runner.cli as cli
import sandbox_runner.environment as env


def test_systemd_purge_timer(monkeypatch, tmp_path):
    containers = tmp_path / "containers.json"
    overlays = tmp_path / "overlays.json"
    overlay_dir = tmp_path / "ovl"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()

    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", containers)
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(containers) + ".lock"))
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))

    env._write_active_containers(["c1", "c2"])
    env._write_active_overlays([str(overlay_dir)])

    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)

    removed = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:2] == ["docker", "ps"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:2] == ["docker", "rm"]:
            removed.append(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    cli.main(["--purge-stale"])

    assert set(removed) == {"c1", "c2"}
    assert containers.exists()
    assert json.loads(containers.read_text()) == []
    assert not overlay_dir.exists()
    assert overlays.exists()
    assert json.loads(overlays.read_text()) == []
