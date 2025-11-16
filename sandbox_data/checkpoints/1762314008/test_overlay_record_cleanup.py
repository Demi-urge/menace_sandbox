import json
import types
from pathlib import Path
import sandbox_runner.environment as env


def test_recorded_overlay_cleanup(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "outside"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()
    file = tmp_path / "overlays.json"
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", file)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(file) + ".lock"))
    env._write_active_overlays([str(overlay_dir)])
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path / "other"))
    monkeypatch.setattr(env, "psutil", None)

    env.purge_leftovers()

    assert not overlay_dir.exists()
    assert file.exists()
    assert json.loads(file.read_text()) == []
