import os
import os
import errno
import sys
import subprocess
import time
from pathlib import Path
import json
from contextlib import suppress

import pytest
import lock_utils
from lock_utils import SandboxLock, Timeout


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
from filelock import FileLock

lock_file = Path(os.environ['LOCK_FILE'])
lock = FileLock(str(lock_file) + '.lock')

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


def test_sandbox_lock_context_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCK_TIMEOUT", "0.2")
    monkeypatch.setattr(lock_utils, "LOCK_TIMEOUT", 0.2)
    monkeypatch.setattr(lock_utils, "is_lock_stale", lambda *a, **k: False)

    attempts: list[int] = [0]

    def _busy_flock(fd: int, flags: int) -> None:
        attempts[0] += 1
        raise BlockingIOError(errno.EAGAIN, "busy")

    monkeypatch.setattr(lock_utils, "flock", _busy_flock)

    lock_path = tmp_path / "context.lock"
    contender = SandboxLock(str(lock_path))

    start = time.perf_counter()
    with pytest.raises(Timeout):
        with contender:
            pass
    elapsed = time.perf_counter() - start

    assert attempts[0] > 1
    assert elapsed >= lock_utils.LOCK_TIMEOUT
    assert elapsed < lock_utils.LOCK_TIMEOUT + 0.5


def test_sandbox_lock_reacquire(tmp_path: Path) -> None:
    lock_path = tmp_path / "reacquire.lock"
    lock = SandboxLock(str(lock_path))

    with lock:
        assert lock.is_locked

    assert not lock.is_locked

    with lock.acquire(timeout=0.1):
        assert lock.is_locked

    assert not lock.is_locked


def test_windows_zero_length_file_lock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lock_utils.os, "name", "nt", raising=False)

    class DummyMSVCRT:
        LK_NBLCK = 0x1

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def locking(self, fd: int, mode: int, length: int) -> None:
            size = os.fstat(fd).st_size
            if size == 0:
                raise PermissionError("cannot lock empty file")
            position = os.lseek(fd, 0, os.SEEK_CUR)
            self.calls.append((fd, mode, length, position))

    dummy = DummyMSVCRT()
    monkeypatch.setattr(lock_utils, "msvcrt", dummy)

    lock_path = tmp_path / "windows.lock"
    lock = SandboxLock(str(lock_path))

    with lock.acquire():
        pass

    assert dummy.calls, "msvcrt.locking should have been invoked"
    _, _, _, position = dummy.calls[0]
    assert position == 0
    assert lock_path.exists()
    assert lock_path.stat().st_size >= 1



STALE_SCRIPT = r"""
import os, time
from pathlib import Path
from filelock import FileLock

lock_file = Path(os.environ['LOCK_FILE'])
lock = FileLock(str(lock_file) + '.lock')

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

class DummyAgent:
    def __init__(self) -> None:
        self.lock_file = Path(os.environ["DUMMY_AGENT_LOCK_FILE"])
        self._global_lock = SandboxLock(str(self.lock_file))
        self.state_file = self.lock_file.with_suffix(".state")
        self.task_queue: list[dict[str, str | None]] = []

    def _persist_state(self) -> None:
        self.state_file.write_text(json.dumps(self.task_queue))

    def _startup_load_state(self) -> None:
        timeout = float(os.environ.get("DUMMY_AGENT_LOCK_TIMEOUT", "1"))
        for _ in range(3):
            try:
                self._global_lock.acquire(timeout=timeout)
                break
            except Timeout:
                if self._global_lock.is_lock_stale(timeout=timeout):
                    with suppress(Exception):
                        os.remove(self.lock_file)
                    continue
                raise
        else:
            raise Timeout(str(self.lock_file))
        try:
            if self.state_file.exists():
                self.task_queue = json.loads(self.state_file.read_text())
        finally:
            self._global_lock.release()


def test_dummy_agent_stale_lock_recovery(monkeypatch, tmp_path):
    lock_path = tmp_path / "agent.lock"
    monkeypatch.setenv("DUMMY_AGENT_LOCK_FILE", str(lock_path))
    monkeypatch.setenv("DUMMY_AGENT_LOCK_TIMEOUT", "1")

    agent = DummyAgent()
    agent.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    agent._persist_state()

    old = time.time() - 2
    lock_path.write_text(f"{os.getpid()+100000},{old}")

    agent2 = DummyAgent()
    agent2._startup_load_state()

    assert not lock_path.exists()
    assert agent2.task_queue[0]["id"] == "a"


def test_dummy_agent_stale_lock_retry(monkeypatch, tmp_path):
    lock_path = tmp_path / "agent.lock"
    monkeypatch.setenv("DUMMY_AGENT_LOCK_FILE", str(lock_path))
    monkeypatch.setenv("DUMMY_AGENT_LOCK_TIMEOUT", "1")

    agent = DummyAgent()
    agent.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    agent._persist_state()

    old = time.time() - 2
    lock_path.write_text(f"{os.getpid()+100000},{old}")

    agent2 = DummyAgent()

    class DummyLock:
        def __init__(self):
            self.calls = 0

        def acquire(self, timeout=0):
            self.calls += 1
            if self.calls == 1:
                raise Timeout(str(lock_path))

        def release(self):
            pass

        def is_lock_stale(self, *a, **k):
            return True

    dummy = DummyLock()
    agent2._global_lock = dummy
    agent2._startup_load_state()

    assert dummy.calls >= 2
    assert not lock_path.exists()
    assert agent2.task_queue[0]["id"] == "a"


def test_dummy_agent_stale_lock_multi_retry(monkeypatch, tmp_path):
    lock_path = tmp_path / "agent.lock"
    monkeypatch.setenv("DUMMY_AGENT_LOCK_FILE", str(lock_path))
    monkeypatch.setenv("DUMMY_AGENT_LOCK_TIMEOUT", "1")

    agent = DummyAgent()
    agent.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    agent._persist_state()

    old = time.time() - 2
    lock_path.write_text(f"{os.getpid()+100000},{old}")

    agent2 = DummyAgent()

    class DummyLock:
        def __init__(self):
            self.calls = 0

        def acquire(self, timeout=0):
            self.calls += 1
            if self.calls < 3:
                raise Timeout(str(lock_path))

        def release(self):
            pass

        def is_lock_stale(self, *a, **k):
            return True

    dummy = DummyLock()
    agent2._global_lock = dummy
    agent2._startup_load_state()

    assert dummy.calls >= 3
    assert not lock_path.exists()
    assert agent2.task_queue[0]["id"] == "a"
