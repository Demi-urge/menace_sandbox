import asyncio
import importlib
import sys
import types
import time
import threading
import socket

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("uvicorn")
import httpx
import uvicorn


@pytest.mark.asyncio
async def test_run_endpoint_busy(monkeypatch):
    """Two concurrent /run requests should return 409 for the second."""
    # Stub heavy optional dependencies before importing the module
    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    filelock_mod = types.ModuleType("filelock")
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
    monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    va = importlib.reload(importlib.import_module("menace_visual_agent_2"))

    # Replace long running pipeline with a short sleep
    def fake_run(prompt: str, branch: str | None = None) -> None:
        time.sleep(0.2)

    monkeypatch.setattr(va, "run_menace_pipeline", fake_run)

    # Replace global file lock with an in-memory lock
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

    monkeypatch.setattr(va, "_global_lock", DummyLock())

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=va.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        task1 = asyncio.create_task(
            client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        )
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(
            client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        )
        resp1, resp2 = await asyncio.gather(task1, task2)

    assert resp1.status_code == 202
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_cancel_queued_task(monkeypatch):
    """A queued task should be cancellable before it starts."""
    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    filelock_mod = types.ModuleType("filelock")
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
    monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    class DummyThread:
        def __init__(self, *a, **k):
            pass
        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", DummyThread)

    va = importlib.reload(importlib.import_module("menace_visual_agent_2"))

    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: None)

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=va.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"}
        )
        tid = resp.json()["id"]
        resp_cancel = await client.post(
            f"/cancel/{tid}", headers={"x-token": va.API_TOKEN}
        )

    assert resp.status_code == 202
    assert resp_cancel.status_code == 202
    assert va.job_status[tid]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_running_task(monkeypatch):
    """Cancelling a running task should fail."""
    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    filelock_mod = types.ModuleType("filelock")
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
    monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    class DummyThread:
        def __init__(self, *a, **k):
            pass
        def start(self):
            pass

    monkeypatch.setattr(threading, "Thread", DummyThread)

    va = importlib.reload(importlib.import_module("menace_visual_agent_2"))
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: None)

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=va.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"}
        )
        tid = resp.json()["id"]

        va.job_status[tid]["status"] = "running"
        va._running_lock.acquire()
        try:
            resp_cancel = await client.post(
                f"/cancel/{tid}", headers={"x-token": va.API_TOKEN}
            )
        finally:
            va._running_lock.release()

    assert resp.status_code == 202
    assert resp_cancel.status_code == 409


@pytest.mark.asyncio
async def test_uvicorn_sequential_requests(monkeypatch):
    """Server should reject overlapping jobs when run via uvicorn."""

    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    filelock_mod = types.ModuleType("filelock")

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
    monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()
    monkeypatch.setenv("MENACE_AGENT_PORT", str(port))

    va = importlib.reload(importlib.import_module("menace_visual_agent_2"))

    def fake_run(prompt: str, branch: str | None = None):
        time.sleep(0.2)

    monkeypatch.setattr(va, "run_menace_pipeline", fake_run)

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

    monkeypatch.setattr(va, "_global_lock", DummyLock())

    config = uvicorn.Config(va.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.01)

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        task1 = asyncio.create_task(
            client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        )
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(
            client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        )
        resp1, resp2 = await asyncio.gather(task1, task2)

    server.should_exit = True
    t.join(timeout=3)

    assert resp1.status_code == 202
    assert resp2.status_code == 409
    assert len(va.job_status) == 1
