import asyncio
import json
import pytest
import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid):
        self.id = f"c{cid}"
        self.status = "running"
    def reload(self):
        pass
    def stop(self, timeout=0):
        pass
    def remove(self, force=True):
        pass

class FailingContainers:
    def __init__(self):
        self.count = 0

    def run(self, *a, **k):
        self.count += 1
        if self.count <= 4:
            raise RuntimeError("fail")
        return DummyContainer(self.count)

class DummyClient:
    def __init__(self):
        self.containers = FailingContainers()


def test_creation_backoff_and_metrics(monkeypatch, tmp_path, caplog):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CREATE_FAILURES.clear()
    env._CONSECUTIVE_CREATE_FAILURES.clear()

    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    monkeypatch.setattr(env, "_CREATE_BACKOFF_BASE", 1.0)
    monkeypatch.setattr(env, "_POOL_METRICS_FILE", tmp_path / "pool.json")
    monkeypatch.setattr(env, "_FAILURE_WARNING_THRESHOLD", 2)

    delays = []
    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(env.asyncio, "sleep", fake_sleep)
    caplog.set_level("WARNING")

    for _ in range(4):
        with pytest.raises(RuntimeError):
            asyncio.run(env._get_pooled_container("img"))

    c, _ = asyncio.run(env._get_pooled_container("img"))
    assert isinstance(c, DummyContainer)
    assert delays == [8.0, 16.0]
    assert "backoff 8.00" in caplog.text
    assert env._CREATE_FAILURES["img"] == 4
    assert env._CONSECUTIVE_CREATE_FAILURES["img"] == 0
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["container_failures_img"] == 4.0
    assert metrics["consecutive_failures_img"] == 0.0
    assert metrics["container_backoff_base"] == 1.0
    data = json.loads((tmp_path / "pool.json").read_text())
    assert data["img"]["failures"] == 4.0
    assert data["img"]["consecutive"] == 0.0
    assert "failing 3 times consecutively" in caplog.text
