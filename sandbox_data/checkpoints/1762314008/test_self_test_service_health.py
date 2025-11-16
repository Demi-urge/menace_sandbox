import asyncio
import json
import socket
import urllib.request
from pathlib import Path

import pytest

from tests.test_self_test_service_async import load_self_test_service, DummyBuilder

sts = load_self_test_service()


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.asyncio
async def test_health_endpoint(monkeypatch, tmp_path: Path) -> None:
    svc = sts.SelfTestService(
        history_path=tmp_path / "hist.json",
        context_builder=DummyBuilder(),
    )

    async def fake_run_once(self) -> None:  # type: ignore[override]
        self.results = {"passed": 2, "failed": 1, "coverage": 100.0, "runtime": 0.05}
        self._store_history({"passed": 2, "failed": 1, "coverage": 100.0, "runtime": 0.05, "ts": "0"})

    monkeypatch.setattr(sts.SelfTestService, "_run_once", fake_run_once)
    port = _free_port()

    loop = asyncio.get_running_loop()
    svc.run_continuous(interval=0.05, loop=loop, health_port=port)
    # wait for server to start
    while svc.health_port is None:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.06)
    data = urllib.request.urlopen(f"http://localhost:{svc.health_port}/health").read().decode()
    info = json.loads(data)
    assert info["passed"] == 2
    assert info["failed"] == 1
    assert info["runtime"] == 0.05
    assert isinstance(info.get("history"), list)
    await svc.stop()
    assert svc._health_thread is None or not svc._health_thread.is_alive()
