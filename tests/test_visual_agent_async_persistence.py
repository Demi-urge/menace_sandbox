import asyncio
import json
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_visual_agent_server import _setup_va


@pytest.mark.asyncio
async def test_parallel_run_requests_persistence(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: None)

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
        tasks = [
            asyncio.create_task(
                client.post(
                    "/run",
                    headers={"x-token": va.API_TOKEN},
                    json={"prompt": str(i)},
                )
            )
            for i in range(3)
        ]
        responses = await asyncio.gather(*tasks)

    success = [r for r in responses if r.status_code == 202]
    failures = [r for r in responses if r.status_code == 409]

    assert len(success) == 3
    assert len(failures) == 0

    data = json.loads((tmp_path / "visual_agent_queue.json").read_text())
    assert len(data["queue"]) == 3
    task_ids = [item["id"] for item in data["queue"]]
    for tid in task_ids:
        assert tid in data["status"]
        assert data["status"][tid]["status"] == "queued"

    va2 = _setup_va(monkeypatch, tmp_path)
    assert list(va2.task_queue) == data["queue"]
    assert va2.job_status == data["status"]
