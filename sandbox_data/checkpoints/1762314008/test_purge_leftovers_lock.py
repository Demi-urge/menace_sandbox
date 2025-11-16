import os
import sys
import json
import subprocess
import time
from pathlib import Path


def test_purge_leftovers_locked(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["ACTIVE_FILE"] = str(tmp_path / "active.json")
    env["OVERLAY_FILE"] = str(tmp_path / "overlays.json")
    env["RECORD_FILE"] = str(tmp_path / "record.txt")
    env["TMPDIR"] = str(tmp_path)
    env["SANDBOX_DOCKER"] = "0"

    Path(env["ACTIVE_FILE"]).write_text(json.dumps(["c1"]))
    Path(env["OVERLAY_FILE"]).write_text("[]")

    script = r"""
import os, json, types, time
from pathlib import Path
import sandbox_runner.environment as env

env._ACTIVE_CONTAINERS_FILE = Path(os.environ['ACTIVE_FILE'])
env._ACTIVE_CONTAINERS_LOCK = env.FileLock(str(env._ACTIVE_CONTAINERS_FILE)+'.lock')
env._ACTIVE_OVERLAYS_FILE = Path(os.environ['OVERLAY_FILE'])
env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(env._ACTIVE_OVERLAYS_FILE)+'.lock')
env.tempfile.gettempdir = lambda: os.environ['TMPDIR']
env._purge_stale_vms = lambda: 0
env.psutil = None
rec = Path(os.environ['RECORD_FILE'])

def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
    if cmd[:3] == ['docker','rm','-f']:
        with rec.open('a') as fh:
            fh.write('rm\n')
    return types.SimpleNamespace(returncode=0, stdout='', stderr='')

env.subprocess.run = fake_run
time.sleep(0.2)
env.purge_leftovers()
"""
    p1 = subprocess.Popen([sys.executable, "-c", script], env=env)
    p2 = subprocess.Popen([sys.executable, "-c", script], env=env)
    p1.wait(timeout=10)
    p2.wait(timeout=10)

    records = Path(env["RECORD_FILE"]).read_text().splitlines()
    assert records == ["rm"]
    assert json.loads(Path(env["ACTIVE_FILE"]).read_text()) == []
