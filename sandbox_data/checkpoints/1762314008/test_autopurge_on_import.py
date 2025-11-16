import json
import os
import subprocess
import sys
from pathlib import Path



def _run_env_script(script: str, env: dict[str, str]):
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_leftover_container_triggers_purge(tmp_path):
    containers = tmp_path / "containers.json"
    overlays = tmp_path / "overlays.json"
    last = tmp_path / "last.txt"
    containers.write_text(json.dumps(["c1"]))
    overlays.write_text("[]")
    last.write_text("0")

    root = Path(__file__).resolve().parents[1]
    parent = root.parent
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(parent), str(root), env.get("PYTHONPATH", "")]),
            "SANDBOX_ACTIVE_CONTAINERS": str(containers),
            "SANDBOX_ACTIVE_OVERLAYS": str(overlays),
            "SANDBOX_LAST_AUTOPURGE": str(last),
            "SANDBOX_AUTOPURGE_THRESHOLD": "1d",
            "TMPDIR": str(tmp_path),
        }
    )

    script = r"""
import json, types, subprocess, sys
calls = []
subprocess.run = lambda *a, **k: (calls.append(a[0]), types.SimpleNamespace(returncode=0, stdout="", stderr=""))[-1]
sys.modules['psutil'] = None
filelock = types.ModuleType('filelock')
class DummyLock:
    def __init__(self, *a, **k):
        pass
    def acquire(self, *a, **k):
        pass
    def release(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
filelock.FileLock = DummyLock
sys.modules['filelock'] = filelock
import sandbox_runner.environment as env
print(json.dumps({
    'active': env._read_active_containers(),
    'calls': calls,
}))
"""

    out = json.loads(_run_env_script(script, env))
    assert out["active"] == []
    assert ["docker", "rm", "-f", "c1"] in out["calls"]
    assert json.loads(containers.read_text()) == []


def test_leftover_overlay_triggers_purge(tmp_path):
    containers = tmp_path / "containers.json"
    overlays = tmp_path / "overlays.json"
    last = tmp_path / "last.txt"
    containers.write_text("[]")
    overlay_dir = tmp_path / "ovl"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()
    overlays.write_text(json.dumps([str(overlay_dir)]))
    last.write_text("0")

    root = Path(__file__).resolve().parents[1]
    parent = root.parent
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(parent), str(root), env.get("PYTHONPATH", "")]),
            "SANDBOX_ACTIVE_CONTAINERS": str(containers),
            "SANDBOX_ACTIVE_OVERLAYS": str(overlays),
            "SANDBOX_LAST_AUTOPURGE": str(last),
            "SANDBOX_AUTOPURGE_THRESHOLD": "1d",
            "TMPDIR": str(tmp_path),
        }
    )

    script = r"""
import json, types, subprocess, sys
calls = []
subprocess.run = lambda *a, **k: (calls.append(a[0]), types.SimpleNamespace(returncode=0, stdout="", stderr=""))[-1]
sys.modules['psutil'] = None
filelock = types.ModuleType('filelock')
class DummyLock:
    def __init__(self, *a, **k):
        pass
    def acquire(self, *a, **k):
        pass
    def release(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
filelock.FileLock = DummyLock
sys.modules['filelock'] = filelock
import sandbox_runner.environment as env
print(json.dumps({
    'overlays': env._read_active_overlays(),
    'calls': calls,
}))
"""

    out = json.loads(_run_env_script(script, env))
    assert out["overlays"] == []
    assert not overlay_dir.exists()
    assert json.loads(overlays.read_text()) == []
