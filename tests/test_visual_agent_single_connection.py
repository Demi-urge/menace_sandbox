import time
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_visual_agent_server import _setup_va


@pytest.mark.asyncio
async def test_run_rejects_when_busy(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path, start_worker=True)
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: time.sleep(0.1))

    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=va.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})
        resp2 = await client.post("/run", headers={"x-token": va.API_TOKEN}, json={"prompt": "p"})

    va._exit_event.set()
    if va._worker_thread:
        va._worker_thread.join(timeout=1)
    if va._autosave_thread:
        va._autosave_thread.join(timeout=1)

    assert resp1.status_code == 202
    assert resp2.status_code == 202
