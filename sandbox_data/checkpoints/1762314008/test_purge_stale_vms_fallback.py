import types
import sandbox_runner.environment as env


def test_purge_stale_vms_removes_untracked_overlay(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "leftover"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()

    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)

    calls = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        calls.append(cmd)
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        if cmd[0] == "kill":
            raise AssertionError("kill should not be called")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    removed = env._purge_stale_vms()

    assert removed >= 1
    assert not overlay_dir.exists()
    assert calls and calls[0][0] == "pgrep"


def test_purge_stale_vms_kills_process_via_pgrep(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "vm"
    overlay_dir.mkdir()
    overlay = overlay_dir / "overlay.qcow2"
    overlay.touch()

    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)

    killed = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:2] == ["pgrep", "-fa"]:
            line = f"123 qemu-system-x86_64 file={overlay}\n"
            return types.SimpleNamespace(returncode=0, stdout=line, stderr="")
        if cmd[0] == "kill":
            killed.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    removed = env._purge_stale_vms()

    assert killed and killed[0][:2] == ["kill", "-9"]
    assert removed >= 1
    assert not overlay_dir.exists()


def test_purge_stale_vms_updates_metrics(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "stale"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()

    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)

    env._STALE_VMS_REMOVED = 0

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    removed = env._purge_stale_vms()

    assert removed >= 1
    assert not overlay_dir.exists()

    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["stale_vms_removed"] >= 1.0
