import asyncio
import importlib
import sys
import types
import time
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


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
