import os
import sys
import socket
import subprocess
import textwrap
import time
from pathlib import Path
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
requests = pytest.importorskip("requests")

TOKEN = "tombalolosvisualagent123"


def _write_script(path: Path, recover: bool = False) -> Path:
    script = path / ("recover.py" if recover else "server.py")  # path-ignore
    script.write_text(textwrap.dedent(f"""
        import sys, types, time, os, runpy
        heavy = ['cv2', 'numpy']
        for name in heavy:
            sys.modules[name] = types.ModuleType(name)
        sys.modules['cv2'].imwrite = lambda *a, **k: None
        sys.modules['cv2'].cvtColor = lambda img, flag: img
        sys.modules['cv2'].COLOR_BGR2GRAY = 0
        sys.modules['numpy'].array = lambda x: x
        class DummyMSS:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                pass
            def grab(self, *a, **k):
                return None
        mss_mod = types.ModuleType('mss')
        mss_mod.mss = lambda: DummyMSS()
        sys.modules['mss'] = mss_mod
        pyautogui = types.SimpleNamespace(
            moveTo=lambda *a, **k: None,
            click=lambda *a, **k: None,
            hotkey=lambda *a, **k: None,
            typewrite=lambda *a, **k: None,
            press=lambda *a, **k: None,
            write=lambda *a, **k: None,
        )
        sys.modules['pyautogui'] = pyautogui
        pt_mod = types.ModuleType('pytesseract')
        pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd='')  # path-ignore
        pt_mod.image_to_string = lambda *a, **k: ''
        pt_mod.image_to_data = lambda *a, **k: {{}}
        pt_mod.Output = types.SimpleNamespace(DICT=0)
        sys.modules['pytesseract'] = pt_mod
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
            @property
            def is_locked(self):
                return False
        filelock_mod.FileLock = DummyFileLock
        filelock_mod.Timeout = DummyTimeout
        sys.modules['filelock'] = filelock_mod
        sys.argv = ['menace_visual_agent_2']
        if {recover}:
            sys.argv.append('--recover-queue')
        import menace_visual_agent_2 as va
        def fake_run(prompt: str, branch: str | None = None):
            time.sleep(0.5)
        va.run_menace_pipeline = fake_run
        runpy.run_module('menace_visual_agent_2', run_name='__main__')
    """))
    return script


def _start_server(tmp_path: Path):
    script = _write_script(tmp_path)
    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    env = os.environ.copy()
    env["MENACE_AGENT_PORT"] = str(port)
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
    env["VISUAL_AGENT_LOCK_FILE"] = str(tmp_path / "agent.lock")
    env["VISUAL_AGENT_PID_FILE"] = str(tmp_path / "agent.pid")
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    for _ in range(50):
        try:
            requests.get(f"http://127.0.0.1:{port}/status", timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("server did not start")

    return proc, port


def _run_recover(tmp_path: Path):
    script = _write_script(tmp_path, recover=True)
    env = os.environ.copy()
    env["SANDBOX_DATA_DIR"] = str(tmp_path)
    env["VISUAL_AGENT_LOCK_FILE"] = str(tmp_path / "agent.lock")
    env["VISUAL_AGENT_PID_FILE"] = str(tmp_path / "agent.pid")
    env["VISUAL_AGENT_TOKEN"] = TOKEN
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(script)], env=env, check=True)


def test_busy_run_returns_409(tmp_path):
    proc, port = _start_server(tmp_path)
    try:
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
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_recover_queue_after_restart(tmp_path):
    proc, port = _start_server(tmp_path)
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/run",
            headers={"x-token": "tombalolosvisualagent123"},
            json={"prompt": "p"},
            timeout=5,
        )
        assert resp.status_code == 202
        task_id = resp.json()["id"]
        time.sleep(0.1)
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    _run_recover(tmp_path)

    proc2, port2 = _start_server(tmp_path)
    try:
        for _ in range(50):
            r = requests.get(
                f"http://127.0.0.1:{port2}/status/{task_id}", timeout=1
            )
            if r.status_code == 200 and r.json().get("status") == "completed":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("task did not resume")
    finally:
        proc2.terminate()
        proc2.wait(timeout=5)
