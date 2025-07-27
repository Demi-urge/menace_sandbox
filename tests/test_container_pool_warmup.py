import asyncio
import types
import sandbox_runner.environment as env


class DummyContainer:
    def __init__(self, cid="c1"):
        self.id = cid
        self.status = "running"
        self.attrs = {"State": {"Health": {"Status": "healthy"}}}

    def reload(self):
        pass

    def stop(self, timeout=0):
        self.stopped = True

    def remove(self, force=True):
        self.removed = True


class DummyClient:
    def __init__(self):
        self.containers = types.SimpleNamespace(run=lambda *a, **k: DummyContainer())


class IncGauge:
    def __init__(self):
        self.called = []

    def labels(self, image):
        def inc():
            self.called.append(image)
        def set_val(v):
            pass
        return types.SimpleNamespace(inc=inc, set=set_val)


def test_ensure_pool_size_async_failure(monkeypatch):
    stub = types.ModuleType("metrics_exporter")
    stub.container_creation_failures_total = IncGauge()
    stub.container_creation_alerts_total = IncGauge()
    stub.container_creation_seconds = IncGauge()
    stub.container_creation_success_total = IncGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CREATE_FAILURES.clear()
    env._CONSECUTIVE_CREATE_FAILURES.clear()

    async def fail_create(image: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(env, "_create_pool_container", fail_create)
    monkeypatch.setattr(env, "_schedule_coroutine", lambda c: asyncio.run(c))

    env._CONTAINER_POOL_SIZE = 1

    env._ensure_pool_size_async("img")

    assert stub.container_creation_failures_total.called == ["img"]
    assert not env._CONTAINER_POOLS.get("img")


def test_cleanup_idle_container_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_METRICS.clear()
    env._STALE_CONTAINERS_REMOVED = 0

    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = 0.0

    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(env.time, "time", lambda: 1.0)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    monkeypatch.setattr(env, "_stop_and_remove", lambda cont: True)

    cleaned, replaced = env._cleanup_idle_containers()

    assert cleaned == 1 and replaced == 0
    assert env._CLEANUP_METRICS["idle"] == 1
    assert env._STALE_CONTAINERS_REMOVED == 1
