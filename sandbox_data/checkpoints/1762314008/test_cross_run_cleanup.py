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


def test_cross_run_cleanup(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["ACTIVE_FILE"] = str(tmp_path / "active.json")
    env["OVERLAY_FILE"] = str(tmp_path / "overlays.json")
    env["TMPDIR"] = str(tmp_path)

    run_script = r"""
import os, sys, time, asyncio, types
from pathlib import Path
import sandbox_runner.environment as env

env._ACTIVE_CONTAINERS_FILE = Path(os.environ['ACTIVE_FILE'])
env._ACTIVE_CONTAINERS_LOCK = env.FileLock(str(env._ACTIVE_CONTAINERS_FILE) + '.lock')
env._ACTIVE_OVERLAYS_FILE = Path(os.environ['OVERLAY_FILE'])
env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(env._ACTIVE_OVERLAYS_FILE) + '.lock')
env.tempfile.gettempdir = lambda: os.environ['TMPDIR']
os.environ['SANDBOX_DOCKER'] = '0'

env._DOCKER_CLIENT = None
env._CONTAINER_POOLS.clear()
env._CONTAINER_DIRS.clear()
env._CONTAINER_LAST_USED.clear()
env._WARMUP_TASKS.clear()
env._CLEANUP_TASK = None
env._CREATE_RETRY_LIMIT = 1

class DummyContainer:
    def __init__(self):
        self.id = 'c1'
    def wait(self):
        time.sleep(60)
        return {'StatusCode': 0}
    def logs(self, stdout=True, stderr=False):
        return b''
    def stats(self, stream=False):
        return {'blkio_stats': {'io_service_bytes_recursive': []},
                'cpu_stats': {'cpu_usage': {'total_usage': 1}},
                'memory_stats': {'max_usage': 1},
                'networks': {}}
    def remove(self):
        pass
    def stop(self, timeout=0):
        pass

class DummyContainers:
    def run(self, image, cmd, **kwargs):
        return DummyContainer()

class DummyClient:
    containers = DummyContainers()

dummy = types.ModuleType('docker')
dummy.from_env = lambda: DummyClient()
dummy.types = types
errors_mod = types.ModuleType('docker.errors')
class DummyErr(Exception):
    pass
errors_mod.DockerException = DummyErr
dummy.errors = errors_mod
sys.modules['docker'] = dummy
sys.modules['docker.errors'] = errors_mod

overlay = Path(os.environ['TMPDIR']) / 'left' / 'overlay.qcow2'
overlay.parent.mkdir(parents=True, exist_ok=True)
overlay.touch()
env._record_active_overlay(str(overlay.parent))
env._record_active_container('c1')
time.sleep(60)
"""
    proc = _run_script(run_script, env)
    try:
        for _ in range(50):
            if Path(env["ACTIVE_FILE"]).exists():
                data = json.loads(Path(env["ACTIVE_FILE"]).read_text() or "[]")
                if data:
                    break
            time.sleep(0.1)
        else:
            proc.kill()
            proc.wait()
            pytest.skip("active container not recorded")

        overlay = Path(env["TMPDIR"]) / "left" / "overlay.qcow2"
        assert overlay.exists()

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()

        data = json.loads(Path(env["ACTIVE_FILE"]).read_text())
        assert data == ['c1']
        overlays = json.loads(Path(env["OVERLAY_FILE"]).read_text())
        assert overlays == [str(overlay.parent)]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    cleanup_script = r"""
import os, sys, json, types
from pathlib import Path
import sandbox_runner.environment as env

env._ACTIVE_CONTAINERS_FILE = Path(os.environ['ACTIVE_FILE'])
env._ACTIVE_CONTAINERS_LOCK = env.FileLock(str(env._ACTIVE_CONTAINERS_FILE) + '.lock')
env._ACTIVE_OVERLAYS_FILE = Path(os.environ['OVERLAY_FILE'])
env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(env._ACTIVE_OVERLAYS_FILE) + '.lock')
env.tempfile.gettempdir = lambda: os.environ['TMPDIR']
os.environ['SANDBOX_DOCKER'] = '0'

env.psutil = None
removed = []

def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
    if cmd[:3] == ['docker', 'rm', '-f']:
        removed.append(cmd[3])
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    return types.SimpleNamespace(returncode=0, stdout='', stderr='')

env.subprocess.run = fake_run
env.purge_leftovers()
print(json.dumps({
    'removed': removed,
    'active': env._read_active_containers(),
    'overlays': env._read_active_overlays(),
}))
"""
    res = subprocess.run([sys.executable, "-c", cleanup_script], env=env, capture_output=True, text=True)
    out = json.loads(res.stdout.strip())
    assert out['removed'] == ['c1']
    assert out['active'] == []
    assert out['overlays'] == []
    assert not (Path(env["TMPDIR"]) / "left").exists()
    assert json.loads(Path(env["ACTIVE_FILE"]).read_text()) == []
    assert json.loads(Path(env["OVERLAY_FILE"]).read_text()) == []
