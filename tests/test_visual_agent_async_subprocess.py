import asyncio
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
pytest.importorskip("httpx")


@pytest.mark.asyncio
async def test_visual_agent_async_subprocess(tmp_path):
    script = tmp_path / "server.py"  # path-ignore
    script.write_text(textwrap.dedent(
        """
        import sys, types, threading, importlib, time, os
        heavy = ['cv2', 'numpy', 'mss', 'pyautogui']
        for name in heavy:
            sys.modules[name] = types.ModuleType(name)
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
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    try:
        for _ in range(50):
            try:
                import httpx
                r = httpx.get(f"http://127.0.0.1:{port}/status")
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            task1 = asyncio.create_task(
                client.post("/run", headers={"x-token": "tombalolosvisualagent123"}, json={"prompt": "p"})
            )
            await asyncio.sleep(0.01)
            task2 = asyncio.create_task(
                client.post("/run", headers={"x-token": "tombalolosvisualagent123"}, json={"prompt": "p"})
            )
            resp1, resp2 = await asyncio.gather(task1, task2)

        codes = {resp1.status_code, resp2.status_code}
        assert codes == {202}
    finally:
        proc.terminate()
        proc.wait(timeout=5)
