import json
import shutil
from pathlib import Path
import sandbox_runner.environment as env


def test_failed_overlay_retry(monkeypatch, tmp_path):
    overlay_dir = tmp_path / "left"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()
    failed_file = tmp_path / "failed.json"
    monkeypatch.setattr(env, "_FAILED_OVERLAYS_FILE", failed_file)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)

    real_rmtree = shutil.rmtree
    calls = {"count": 0}

    def fake_rmtree(path, *a, **k):
        if Path(path) == overlay_dir and calls["count"] == 0:
            calls["count"] += 1
            raise PermissionError("locked")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(env.shutil, "rmtree", fake_rmtree)

    env.purge_leftovers()

    assert overlay_dir.exists()
    assert failed_file.exists()
    assert json.loads(failed_file.read_text()) == [str(overlay_dir)]

    monkeypatch.setattr(env.shutil, "rmtree", real_rmtree)
    env.purge_leftovers()

    assert not overlay_dir.exists()
    assert json.loads(failed_file.read_text()) == []
