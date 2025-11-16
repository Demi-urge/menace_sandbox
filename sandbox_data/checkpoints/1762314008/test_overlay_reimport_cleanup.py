import json
import os
import subprocess
import sys
from pathlib import Path


def test_overlay_cleanup_on_reimport(tmp_path):
    overlay_dir = tmp_path / "leftover"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()
    active_file = tmp_path / "active.json"
    failed_file = tmp_path / "failed.json"
    active_file.write_text(json.dumps([str(overlay_dir)]))
    failed_file.write_text("[]")

    root = Path(__file__).resolve().parents[1]
    parent = root.parent
    env = os.environ.copy()
    env["SANDBOX_ACTIVE_OVERLAYS"] = str(active_file)
    env["SANDBOX_FAILED_OVERLAYS"] = str(failed_file)
    env["TMPDIR"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join([str(parent), str(root), env.get("PYTHONPATH", "")])
    script = "import sandbox_runner.environment as e; e.purge_leftovers()"
    subprocess.run([sys.executable, "-c", script], env=env, check=True)

    assert not overlay_dir.exists()
    assert json.loads(active_file.read_text()) == []
    assert json.loads(failed_file.read_text()) == []
