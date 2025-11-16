import os
import sys
import json
import signal
import subprocess
import time
from pathlib import Path

import pytest


def _run_script(script: str, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", script], env=env)


def test_cross_run_qemu_overlay_cleanup(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["OVERLAY_FILE"] = str(tmp_path / "overlays.json")
    env["TMPDIR"] = str(tmp_path)
    env["PID_FILE"] = str(tmp_path / "pid.txt")

    run_script = r"""
import os, time, shutil, subprocess
from pathlib import Path
import sandbox_runner.environment as env

env._ACTIVE_OVERLAYS_FILE = Path(os.environ['OVERLAY_FILE'])
env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(env._ACTIVE_OVERLAYS_FILE) + '.lock')
env.tempfile.gettempdir = lambda: os.environ['TMPDIR']
os.environ['SANDBOX_DOCKER'] = '0'

qemu = Path(os.environ['TMPDIR']) / 'qemu-system-x86_64'
os.symlink(shutil.which('sleep'), qemu)
overlay = Path(os.environ['TMPDIR']) / 'vm' / 'overlay.qcow2'
overlay.parent.mkdir(parents=True, exist_ok=True)
overlay.touch()
env._record_active_overlay(str(overlay.parent))
proc = subprocess.Popen([str(qemu), '60', f'file={overlay}'])
Path(os.environ['PID_FILE']).write_text(str(proc.pid))
time.sleep(60)
"""
    proc = _run_script(run_script, env)
    try:
        for _ in range(50):
            if Path(env["OVERLAY_FILE"]).exists() and Path(env["PID_FILE"]).exists():
                data = json.loads(Path(env["OVERLAY_FILE"]).read_text() or "[]")
                if data:
                    break
            time.sleep(0.1)
        else:
            proc.kill()
            proc.wait()
            pytest.skip("active overlay not recorded")

        overlay_dir = Path(env["TMPDIR"]) / "vm"
        assert overlay_dir.exists()

        qemu_pid = int(Path(env["PID_FILE"]).read_text())
        os.kill(qemu_pid, signal.SIGKILL)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()

        data = json.loads(Path(env["OVERLAY_FILE"]).read_text())
        assert data == [str(overlay_dir)]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    cleanup_script = r"""
import os, sys, json, types
from pathlib import Path
import sandbox_runner.environment as env

env._ACTIVE_OVERLAYS_FILE = Path(os.environ['OVERLAY_FILE'])
env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(env._ACTIVE_OVERLAYS_FILE) + '.lock')
env.tempfile.gettempdir = lambda: os.environ['TMPDIR']
os.environ['SANDBOX_DOCKER'] = '0'

env.psutil = None

def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
    return types.SimpleNamespace(returncode=0, stdout='', stderr='')

env.subprocess.run = fake_run
env.purge_leftovers()
print(json.dumps({'overlays': env._read_active_overlays()}))
"""
    res = subprocess.run([sys.executable, "-c", cleanup_script], env=env, capture_output=True, text=True)
    out = json.loads(res.stdout.strip())
    assert out['overlays'] == []
    assert not (Path(env["TMPDIR"]) / "vm").exists()
    assert json.loads(Path(env["OVERLAY_FILE"]).read_text()) == []
