import os
import time
import types
import sandbox_runner.environment as env


def test_old_overlay_cleanup(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "old"
    overlay_dir.mkdir()
    overlay = overlay_dir / "overlay.qcow2"
    overlay.touch()
    old = time.time() - 10
    os.utime(overlay, (old, old))
    os.utime(overlay_dir, (old, old))

    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 5.0)

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    removed = env._purge_stale_vms()
    assert removed >= 1
    assert not overlay_dir.exists()

