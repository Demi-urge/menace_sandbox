import os
import subprocess
import psutil
import shutil
import sandbox_runner.environment as env

def test_purge_leftovers_kills_qemu(monkeypatch, tmp_path):
    qemu = tmp_path / "qemu-system-x86_64"
    os.symlink(shutil.which("sleep"), qemu)
    proc = subprocess.Popen([str(qemu), "60"])  # long running dummy
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)
    (tmp_path / "left").mkdir()
    (tmp_path / "left" / "overlay.qcow2").touch()
    try:
        assert psutil.pid_exists(proc.pid)
        env.purge_leftovers()
        assert proc.poll() is not None
        assert not (tmp_path / "left").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_purge_leftovers_kills_qemu_subprocess(monkeypatch, tmp_path):
    script = tmp_path / "qemu-system-x86_64"
    script.write_text("#!/bin/sh\nsleep \"$1\"\n")
    script.chmod(0o755)
    overlay_dir = tmp_path / "left2"
    overlay_dir.mkdir()
    overlay = overlay_dir / "overlay.qcow2"
    overlay.touch()
    proc = subprocess.Popen([str(script), "60", f"file={overlay}"])
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)
    try:
        env.purge_leftovers()
        assert proc.poll() is not None
        assert not overlay_dir.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
