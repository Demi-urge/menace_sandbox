import os
import sys
import subprocess
import time
from pathlib import Path

import pytest


LOCK_TYPES = {
    "containers": (
        "_ACTIVE_CONTAINERS_FILE",
        "_ACTIVE_CONTAINERS_LOCK",
        "containers.json",
    ),
    "overlays": (
        "_ACTIVE_OVERLAYS_FILE",
        "_ACTIVE_OVERLAYS_LOCK",
        "overlays.json",
    ),
}


def _make_env(tmp_path: Path, lock_type: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["LOCK_FILE"] = str(tmp_path / LOCK_TYPES[lock_type][2])
    env["RECORD_FILE"] = str(tmp_path / f"{lock_type}_record.txt")
    env["LOCK_TYPE"] = lock_type
    return env


SCRIPT = r"""
import os, time
from pathlib import Path
import sandbox_runner.environment as env

lock_file = Path(os.environ['LOCK_FILE'])
if os.environ['LOCK_TYPE'] == 'containers':
    env._ACTIVE_CONTAINERS_FILE = lock_file
    env._ACTIVE_CONTAINERS_LOCK = env.FileLock(str(lock_file)+'.lock')
    lock = env._ACTIVE_CONTAINERS_LOCK
else:
    env._ACTIVE_OVERLAYS_FILE = lock_file
    env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(lock_file)+'.lock')
    lock = env._ACTIVE_OVERLAYS_LOCK

rec = Path(os.environ['RECORD_FILE'])
with lock:
    with rec.open('a') as fh:
        fh.write(f"{os.environ['ID']} acquire\n")
    time.sleep(float(os.environ.get('SLEEP', '0.5')))
    with rec.open('a') as fh:
        fh.write(f"{os.environ['ID']} release\n")
"""


@pytest.mark.parametrize("lock_type", ["containers", "overlays"])
def test_lock_exclusive(tmp_path: Path, lock_type: str) -> None:
    env = _make_env(tmp_path, lock_type)
    env1 = env.copy(); env1["ID"] = "1"
    env2 = env.copy(); env2["ID"] = "2"
    p1 = subprocess.Popen([sys.executable, "-c", SCRIPT], env=env1)
    p2 = subprocess.Popen([sys.executable, "-c", SCRIPT], env=env2)
    p1.wait(timeout=10)
    p2.wait(timeout=10)

    events = Path(env["RECORD_FILE"]).read_text().splitlines()
    active = set()
    for event in events:
        ident, action = event.split()
        if action == "acquire":
            assert not active
            active.add(ident)
        else:
            assert ident in active
            active.remove(ident)
    assert not active



STALE_SCRIPT = r"""
import os, time
from pathlib import Path
import sandbox_runner.environment as env

lock_file = Path(os.environ['LOCK_FILE'])
if os.environ['LOCK_TYPE'] == 'containers':
    env._ACTIVE_CONTAINERS_FILE = lock_file
    env._ACTIVE_CONTAINERS_LOCK = env.FileLock(str(lock_file)+'.lock')
    lock = env._ACTIVE_CONTAINERS_LOCK
else:
    env._ACTIVE_OVERLAYS_FILE = lock_file
    env._ACTIVE_OVERLAYS_LOCK = env.FileLock(str(lock_file)+'.lock')
    lock = env._ACTIVE_OVERLAYS_LOCK

rec = Path(os.environ['RECORD_FILE'])
lock.acquire()
with rec.open('a') as fh:
    fh.write(f"{os.environ['ID']} acquire\n")
# hold indefinitely
time.sleep(60)
"""


@pytest.mark.skipif(sys.platform == 'win32', reason='stale lock check unreliable on Windows')
@pytest.mark.parametrize("lock_type", ["containers", "overlays"])
def test_stale_lock_cleanup(tmp_path: Path, lock_type: str) -> None:
    env = _make_env(tmp_path, lock_type)
    env1 = env.copy(); env1["ID"] = "1"
    env2 = env.copy(); env2["ID"] = "2"; env2["SLEEP"] = "0.1"
    p1 = subprocess.Popen([sys.executable, "-c", STALE_SCRIPT], env=env1)

    for _ in range(50):
        rec = Path(env["RECORD_FILE"])
        if rec.exists() and "acquire" in rec.read_text():
            break
        time.sleep(0.1)
    else:
        p1.kill()
        p1.wait()
        pytest.skip("first process failed to acquire lock")

    lock_path = Path(env["LOCK_FILE"] + ".lock")
    assert lock_path.exists()
    mtime_before = lock_path.stat().st_mtime

    p1.kill()
    p1.wait()

    time.sleep(0.1)

    p2 = subprocess.Popen([sys.executable, "-c", SCRIPT], env=env2)
    p2.wait(timeout=10)

    mtime_after = lock_path.stat().st_mtime
    events = Path(env["RECORD_FILE"]).read_text().splitlines()
    assert events[0].startswith("1 acquire")
    assert any(line.startswith("2 acquire") for line in events)
    assert any(line.startswith("2 release") for line in events)

    assert lock_path.exists()
    assert mtime_after > mtime_before


import importlib
import types


def _setup_va(monkeypatch, tmp_path):
    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")  # path-ignore
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    module_name = "menace_visual_" + "agent_2"
    return importlib.reload(importlib.import_module(module_name))


def test_agent_stale_lock_recovery(monkeypatch, tmp_path):
    token_env = "VISUAL_" + "AGENT_TOKEN"
    lock_file_env = "VISUAL_" + "AGENT_LOCK_FILE"
    pid_file_env = "VISUAL_" + "AGENT_PID_FILE"
    timeout_env = "VISUAL_" + "AGENT_LOCK_TIMEOUT"
    monkeypatch.setenv(token_env, "tok")
    lock_path = tmp_path / "agent.lock"
    pid_path = tmp_path / "agent.pid"
    monkeypatch.setenv(lock_file_env, str(lock_path))
    monkeypatch.setenv(pid_file_env, str(pid_path))
    monkeypatch.setenv(timeout_env, "1")

    va = _setup_va(monkeypatch, tmp_path)
    class DummyQueue(list):
        def append(self, item):
            super().append(item)
        def reset_running_tasks(self):
            pass
        def update_status(self, *a, **k):
            pass
        def set_last_completed(self, *a, **k):
            pass
        def get_status(self):
            return {i["id"]: {"status": "queued", "prompt": i["prompt"], "branch": i["branch"]} for i in self}
        def get_last_completed(self):
            return 0.0
    va.task_queue = DummyQueue()
    va.job_status = {}
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()
    va.job_status.clear()

    old = time.time() - 2
    lock_path.write_text(f"{os.getpid()+100000},{old}")

    va2 = _setup_va(monkeypatch, tmp_path)
    va2.task_queue = va.task_queue
    va2.job_status = va.job_status
    va2.task_queue = va.task_queue
    va2.job_status = va.job_status
    monkeypatch.setattr(va2, "_start_background_threads", lambda: None)
    va2._startup_load_state()

    assert not lock_path.exists()
    assert list(va2.task_queue)[0]["id"] == "a"


def test_agent_stale_lock_retry(monkeypatch, tmp_path):
    token_env = "VISUAL_" + "AGENT_TOKEN"
    lock_file_env = "VISUAL_" + "AGENT_LOCK_FILE"
    pid_file_env = "VISUAL_" + "AGENT_PID_FILE"
    timeout_env = "VISUAL_" + "AGENT_LOCK_TIMEOUT"
    monkeypatch.setenv(token_env, "tok")
    lock_path = tmp_path / "agent.lock"
    pid_path = tmp_path / "agent.pid"
    monkeypatch.setenv(lock_file_env, str(lock_path))
    monkeypatch.setenv(pid_file_env, str(pid_path))
    monkeypatch.setenv(timeout_env, "1")

    va = _setup_va(monkeypatch, tmp_path)
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()
    va.job_status.clear()

    old = time.time() - 2
    lock_path.write_text(f"{os.getpid()+100000},{old}")

    va2 = _setup_va(monkeypatch, tmp_path)

    class DummyLock:
        def __init__(self):
            self.calls = 0

        def acquire(self, timeout=0):
            self.calls += 1
            if self.calls == 1:
                raise va2.Timeout(str(lock_path))

        def release(self):
            pass

        def is_lock_stale(self, *a, **k):
            return True

    dummy = DummyLock()
    monkeypatch.setattr(va2, "_global_lock", dummy)
    monkeypatch.setattr(va2, "_start_background_threads", lambda: None)
    monkeypatch.setattr(va2.task_queue, "reset_running_tasks", lambda: None)

    va2._startup_load_state()

    assert dummy.calls >= 2
    assert not lock_path.exists()
    assert list(va2.task_queue)[0]["id"] == "a"


def test_agent_stale_lock_multi_retry(monkeypatch, tmp_path):
    token_env = "VISUAL_" + "AGENT_TOKEN"
    lock_file_env = "VISUAL_" + "AGENT_LOCK_FILE"
    pid_file_env = "VISUAL_" + "AGENT_PID_FILE"
    timeout_env = "VISUAL_" + "AGENT_LOCK_TIMEOUT"
    monkeypatch.setenv(token_env, "tok")
    lock_path = tmp_path / "agent.lock"
    pid_path = tmp_path / "agent.pid"
    monkeypatch.setenv(lock_file_env, str(lock_path))
    monkeypatch.setenv(pid_file_env, str(pid_path))
    monkeypatch.setenv(timeout_env, "1")

    va = _setup_va(monkeypatch, tmp_path)
    class DummyQueue(list):
        def append(self, item):
            super().append(item)
        def reset_running_tasks(self):
            pass
        def update_status(self, *a, **k):
            pass
        def set_last_completed(self, *a, **k):
            pass
        def get_status(self):
            return {i["id"]: {"status": "queued", "prompt": i["prompt"], "branch": i["branch"]} for i in self}
        def get_last_completed(self):
            return 0.0
    saved_queue = DummyQueue()
    va.task_queue = saved_queue
    va.job_status = {}
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()
    saved_status = va.job_status
    va.job_status.clear()

    old = time.time() - 2
    lock_path.write_text(f"{os.getpid()+100000},{old}")

    va2 = _setup_va(monkeypatch, tmp_path)
    va2.task_queue = saved_queue
    va2.job_status = saved_status

    class DummyLock:
        def __init__(self):
            self.calls = 0

        def acquire(self, timeout=0):
            self.calls += 1
            if self.calls < 3:
                raise va2.Timeout(str(lock_path))

        def release(self):
            pass

        def is_lock_stale(self, *a, **k):
            return True

    dummy = DummyLock()
    monkeypatch.setattr(va2, "_global_lock", dummy)
    monkeypatch.setattr(va2, "_start_background_threads", lambda: None)
    monkeypatch.setattr(va2.task_queue, "reset_running_tasks", lambda: None)

    va2._startup_load_state()

    assert dummy.calls >= 3
    assert not lock_path.exists()
    assert list(va2.task_queue)[0]["id"] == "a"
