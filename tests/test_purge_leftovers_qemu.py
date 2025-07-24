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
    (tmp_path / "left" ).mkdir()
    (tmp_path / "left" / "overlay.qcow2").touch()
    env._STALE_VMS_REMOVED = 0
    env._VM_OVERLAYS_REMOVED = 0
    try:
        assert psutil.pid_exists(proc.pid)
        env.purge_leftovers()
        assert proc.poll() is not None
        assert not (tmp_path / "left").exists()
        assert env._VM_OVERLAYS_REMOVED == 1
        metrics = env.collect_metrics(0.0, 0.0, None)
        assert metrics["vm_overlays_removed"] == 1.0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
