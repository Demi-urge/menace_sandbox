import json
import types
import shutil
from pathlib import Path
import sandbox_runner.environment as env


def test_windows_locked_overlay_retry(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "ov"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()

    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", tmp_path / "overlays.json")
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(tmp_path / "overlays.json") + ".lock"))
    (tmp_path / "overlays.json").write_text(json.dumps([str(overlay_dir)]))
    monkeypatch.setattr(env, "_FAILED_OVERLAYS_FILE", tmp_path / "failed.json")
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "os", types.SimpleNamespace(name="nt"), raising=False)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)

    real_rmtree = shutil.rmtree

    def fake_rmtree(path, ignore_errors=False):
        if Path(path) == overlay_dir:
            raise PermissionError("locked")
        return real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(env.shutil, "rmtree", fake_rmtree)

    calls = []

    def helper(path, attempts=5, base=0.2):
        calls.append(path)
        real_rmtree(path)
        return True

    monkeypatch.setattr(env, "_rmtree_windows", helper)

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:2] == ["pgrep", "-fa"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    removed = env._purge_stale_vms()

    assert removed >= 1
    assert not overlay_dir.exists()
    assert calls and calls[0] == str(overlay_dir)
