import asyncio
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_visual_agent_server import _setup_va


@pytest.mark.asyncio
async def test_run_busy_when_queued(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    calls = {"n": 0}
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=va.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        resp2 = await client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})

    assert resp1.status_code == 202
    assert resp2.status_code == 409
    assert len(va.task_queue) == 1
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_run_busy_when_running(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path, start_worker=True)
    calls = {"n": 0}
    started = threading.Event()

    def fake_run(prompt: str, branch: str | None = None):
        calls["n"] += 1
        started.set()
        time.sleep(0.1)

    monkeypatch.setattr(va, "run_menace_pipeline", fake_run)

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=va.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        started.wait(1)
        resp2 = await client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})

    va._exit_event.set()
    va._worker_thread.join(timeout=1)
    va._autosave_thread.join(timeout=1)

    assert resp1.status_code == 202
    assert resp2.status_code == 409
    assert calls["n"] == 1
    assert not va.task_queue
