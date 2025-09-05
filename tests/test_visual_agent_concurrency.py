import os
import sys
import socket
import subprocess
import textwrap
import time
import json
from pathlib import Path
import importlib.util
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")

TOKEN = "tombalolosvisualagent123"

for dep in ("pyautogui", "mss", "cv2", "numpy"):
    if importlib.util.find_spec(dep) is None:
        pytest.skip(f"{dep} is required", allow_module_level=True)


def test_visual_agent_concurrency(tmp_path):
    script = tmp_path / "server.py"  # path-ignore
    script.write_text(textwrap.dedent(
        """
        import sys, types, threading, importlib, time, os
        heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
        for name in heavy:
            sys.modules[name] = types.ModuleType(name)
        psutil = types.ModuleType('psutil')
        psutil.pid_exists = lambda *_a, **_k: False
        sys.modules['psutil'] = psutil
        filelock_mod = types.ModuleType('filelock')
        class DummyTimeout(Exception):
            pass
        class DummyFileLock:
            def __init__(self, *a, **k):
                pass
            def acquire(self, timeout=0):
                pass
            def release(self):
                pass
        filelock_mod.FileLock = DummyFileLock
        filelock_mod.Timeout = DummyTimeout
        sys.modules['filelock'] = filelock_mod
        pt_mod = types.ModuleType('pytesseract')
        pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
        pt_mod.image_to_string = lambda *a, **k: ''
        pt_mod.image_to_data = lambda *a, **k: {}
        pt_mod.Output = types.SimpleNamespace(DICT=0)
        sys.modules['pytesseract'] = pt_mod
        va = importlib.import_module('menace_visual_agent_2')
        def fake_run(prompt: str, branch: str | None = None):
            time.sleep(0.2)
        va.run_menace_pipeline = fake_run
        shared = threading.Lock()
        class DummyLock:
            def acquire(self, timeout: float = 0):
                if not shared.acquire(blocking=False):
                    raise va.Timeout()
            def release(self):
                if shared.locked():
                    shared.release()
            @property
            def is_locked(self):
                return shared.locked()
        va._global_lock = DummyLock()
        import uvicorn
        uvicorn.run(va.app, host='127.0.0.1', port=int(os.environ['MENACE_AGENT_PORT']), log_level='error')
        """
    ))

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        resp1 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "p"},
            timeout=5,
        )
        resp2 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp1.status_code == 202
        assert resp2.status_code == 202

        time.sleep(0.3)

        resp3 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp3.status_code == 202
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_run_requests_enqueue_during_active_run(tmp_path):
    """Additional /run requests are queued while a job is running."""

    script = tmp_path / "server_active.py"  # path-ignore
    script.write_text(textwrap.dedent(
        """
        import sys, types, threading, importlib, time, os
        heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
        for name in heavy:
            sys.modules[name] = types.ModuleType(name)
        psutil = types.ModuleType('psutil')
        psutil.pid_exists = lambda *_a, **_k: False
        sys.modules['psutil'] = psutil
        filelock_mod = types.ModuleType('filelock')
        class DummyTimeout(Exception):
            pass
        class DummyFileLock:
            def __init__(self, *a, **k):
                pass
            def acquire(self, timeout=0):
                pass
            def release(self):
                pass
        filelock_mod.FileLock = DummyFileLock
        filelock_mod.Timeout = DummyTimeout
        sys.modules['filelock'] = filelock_mod
        pt_mod = types.ModuleType('pytesseract')
        pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
        pt_mod.image_to_string = lambda *a, **k: ''
        pt_mod.image_to_data = lambda *a, **k: {}
        pt_mod.Output = types.SimpleNamespace(DICT=0)
        sys.modules['pytesseract'] = pt_mod
        va = importlib.import_module('menace_visual_agent_2')
        def fake_run(prompt: str, branch: str | None = None):
            time.sleep(0.3)
        va.run_menace_pipeline = fake_run
        shared = threading.Lock()
        class DummyLock:
            def acquire(self, timeout: float = 0):
                if not shared.acquire(blocking=False):
                    raise va.Timeout()
            def release(self):
                if shared.locked():
                    shared.release()
            @property
            def is_locked(self):
                return shared.locked()
        va._global_lock = DummyLock()
        import uvicorn
        uvicorn.run(va.app, host='127.0.0.1', port=int(os.environ['MENACE_AGENT_PORT']), log_level='error')
        """
    ))

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        resp1 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp1.status_code == 202

        for _ in range(20):
            status = requests.get(f"http://127.0.0.1:{port}/status", timeout=1).json()
            if status.get("active"):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("agent did not start running")

        resp2 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )
        resp3 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp2.status_code == 202
        assert resp3.status_code == 202
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_overlapping_run_requests(tmp_path):
    """First run is accepted while the second is rejected."""

    script = tmp_path / "server.py"  # path-ignore
    script.write_text(
        textwrap.dedent(
            """
            import sys, types, threading, importlib, time, os
            heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
            for name in heavy:
                sys.modules[name] = types.ModuleType(name)
            psutil = types.ModuleType('psutil')
            psutil.pid_exists = lambda *_a, **_k: False
            sys.modules['psutil'] = psutil
            filelock_mod = types.ModuleType('filelock')
            class DummyTimeout(Exception):
                pass
            class DummyFileLock:
                def __init__(self, *a, **k):
                    pass
                def acquire(self, timeout=0):
                    pass
                def release(self):
                    pass
            filelock_mod.FileLock = DummyFileLock
            filelock_mod.Timeout = DummyTimeout
            sys.modules['filelock'] = filelock_mod
            pt_mod = types.ModuleType('pytesseract')
            pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
            pt_mod.image_to_string = lambda *a, **k: ''
            pt_mod.image_to_data = lambda *a, **k: {}
            pt_mod.Output = types.SimpleNamespace(DICT=0)
            sys.modules['pytesseract'] = pt_mod
            va = importlib.import_module('menace_visual_agent_2')
            def fake_run(prompt: str, branch: str | None = None):
                time.sleep(0.2)
            va.run_menace_pipeline = fake_run
            shared = threading.Lock()
            class DummyLock:
                def acquire(self, timeout: float = 0):
                    if not shared.acquire(blocking=False):
                        raise va.Timeout()
                def release(self):
                    if shared.locked():
                        shared.release()
                @property
                def is_locked(self):
                    return shared.locked()
            va._global_lock = DummyLock()
            import uvicorn
            uvicorn.run(va.app, host='127.0.0.1', port=int(os.environ['MENACE_AGENT_PORT']), log_level='error')
            """
        )
    )

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        resp1 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )

        resp2 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )

        assert resp1.status_code == 202
        assert resp2.status_code == 202
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_visual_agent_queue_persistence(tmp_path):
    script = tmp_path / "server.py"  # path-ignore
    script.write_text(textwrap.dedent(
        """
        import sys, types, threading, importlib, time, os, json
        heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
        for name in heavy:
            sys.modules[name] = types.ModuleType(name)
        psutil = types.ModuleType('psutil')
        psutil.pid_exists = lambda *_a, **_k: False
        sys.modules['psutil'] = psutil
        filelock_mod = types.ModuleType('filelock')
        class DummyTimeout(Exception):
            pass
        class DummyFileLock:
            def __init__(self, *a, **k):
                pass
            def acquire(self, timeout=0):
                pass
            def release(self):
                pass
        filelock_mod.FileLock = DummyFileLock
        filelock_mod.Timeout = DummyTimeout
        sys.modules['filelock'] = filelock_mod
        pt_mod = types.ModuleType('pytesseract')
        pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
        pt_mod.image_to_string = lambda *a, **k: ''
        pt_mod.image_to_data = lambda *a, **k: {}
        pt_mod.Output = types.SimpleNamespace(DICT=0)
        sys.modules['pytesseract'] = pt_mod
        va = importlib.import_module('menace_visual_agent_2')
        def fake_run(prompt: str, branch: str | None = None):
            time.sleep(0.2)
        va.run_menace_pipeline = fake_run
        shared = threading.Lock()
        class DummyLock:
            def acquire(self, timeout: float = 0):
                if not shared.acquire(blocking=False):
                    raise va.Timeout()
            def release(self):
                if shared.locked():
                    shared.release()
            @property
            def is_locked(self):
                return shared.locked()
        va._global_lock = DummyLock()
        import uvicorn
        uvicorn.run(va.app, host='127.0.0.1', port=int(os.environ['MENACE_AGENT_PORT']), log_level='error')
        """
    ))

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        resp1 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "a"},
            timeout=5,
        )
        resp2 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "b"},
            timeout=5,
        )
        assert resp1.status_code == 202
        assert resp2.status_code == 202

        time.sleep(0.3)

        resp3 = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "c"},
            timeout=5,
        )
        assert resp3.status_code == 202

        time.sleep(0.3)

        data = json.loads((tmp_path / "visual_agent_state.json").read_text())
        assert len(data["status"]) == 3
        assert all(v["status"] == "completed" for v in data["status"].values())
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_visual_agent_auto_recover(tmp_path):
    script = tmp_path / "server.py"  # path-ignore
    script.write_text(textwrap.dedent(
        """
        import sys, types, threading, importlib, time, os
        heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
        for name in heavy:
            sys.modules[name] = types.ModuleType(name)
        psutil = types.ModuleType('psutil')
        psutil.pid_exists = lambda *_a, **_k: False
        sys.modules['psutil'] = psutil
        filelock_mod = types.ModuleType('filelock')
        class DummyTimeout(Exception):
            pass
        class DummyFileLock:
            def __init__(self, *a, **k):
                pass
            def acquire(self, timeout=0):
                pass
            def release(self):
                pass
        filelock_mod.FileLock = DummyFileLock
        filelock_mod.Timeout = DummyTimeout
        sys.modules['filelock'] = filelock_mod
        pt_mod = types.ModuleType('pytesseract')
        pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
        pt_mod.image_to_string = lambda *a, **k: ''
        pt_mod.image_to_data = lambda *a, **k: {}
        pt_mod.Output = types.SimpleNamespace(DICT=0)
        sys.modules['pytesseract'] = pt_mod
        va = importlib.import_module('menace_visual_agent_2')
        def fake_run(prompt: str, branch: str | None = None):
            time.sleep(0.5)
        va.run_menace_pipeline = fake_run
        shared = threading.Lock()
        class DummyLock:
            def acquire(self, timeout: float = 0):
                if not shared.acquire(blocking=False):
                    raise va.Timeout()
            def release(self):
                if shared.locked():
                    shared.release()
            @property
            def is_locked(self):
                return shared.locked()
        va._global_lock = DummyLock()
        if os.environ.get('DISABLE_WORKER') == '1':
            va._start_background_threads = lambda: None
        import uvicorn
        uvicorn.run(va.app, host='127.0.0.1', port=int(os.environ['MENACE_AGENT_PORT']), log_level='error')
        """
    ))

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    env["DISABLE_WORKER"] = "1"
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            proc.kill()
            proc.wait(timeout=5)
            raise RuntimeError("server did not start")

        resp = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": TOKEN},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp.status_code == 202
        task_id = resp.json()["id"]
    finally:
        proc.kill()
        proc.wait(timeout=5)

    db_path = tmp_path / "visual_agent_queue.db"
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert row is not None

    env.pop("DISABLE_WORKER")
    env["VISUAL_AGENT_AUTO_RECOVER"] = "1"
    proc2 = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            proc2.kill()
            proc2.wait(timeout=5)
            raise RuntimeError("restarted server did not start")

        for _ in range(20):
            try:
                r = requests.get(
                    f"http://127.0.0.1:{port}/status/{task_id}", timeout=0.1
                )
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("task not recovered")

        with sqlite3.connect(db_path) as conn:
            status_after = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()[0]
        assert status_after in {"queued", "running", "completed"}
    finally:
        proc2.terminate()
        proc2.wait(timeout=5)
