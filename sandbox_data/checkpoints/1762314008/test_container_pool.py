import asyncio
import sandbox_runner.environment as env
import pytest

class DummyContainer:
    def __init__(self, cid):
        self.id = f"c{cid}"
        self.status = "running"
        self.stopped = False
        self.removed = False
        self.attrs = {}

    def reload(self):
        pass

    def stop(self, timeout=0):
        self.stopped = True

    def remove(self, force=True):
        self.removed = True

class DummyContainers:
    def __init__(self):
        self.count = 0
    def run(self, *a, **k):
        self.count += 1
        return DummyContainer(self.count)

class DummyClient:
    def __init__(self):
        self.containers = DummyContainers()


def _reset(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CREATE_FAILURES.clear()
    env._CONSECUTIVE_CREATE_FAILURES.clear()
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)

def test_reuse_and_unhealthy_cleanup(monkeypatch, tmp_path):
    _reset(monkeypatch)
    created = []

    async def create(image: str):
        c = DummyContainer(len(created))
        td = tmp_path / c.id
        td.mkdir()
        created.append(c)
        env._CONTAINER_DIRS[c.id] = str(td)
        env._CONTAINER_LAST_USED[c.id] = env.time.time()
        env._CONTAINER_CREATED[c.id] = env.time.time()
        return c, str(td)

    monkeypatch.setattr(env, "_create_pool_container", create)

    c1, td1 = asyncio.run(env._get_pooled_container("img"))
    env._release_container("img", c1)

    c2, td2 = asyncio.run(env._get_pooled_container("img"))
    assert c2 is c1
    assert td2 == td1
    env._release_container("img", c2)

    c1.status = "exited"
    env._CONTAINER_POOLS["img"] = [c1]

    new = DummyContainer("new")

    async def create_new(image: str):
        td = tmp_path / "new"
        td.mkdir()
        env._CONTAINER_DIRS[new.id] = str(td)
        env._CONTAINER_LAST_USED[new.id] = env.time.time()
        env._CONTAINER_CREATED[new.id] = env.time.time()
        return new, str(td)

    monkeypatch.setattr(env, "_create_pool_container", create_new)

    c3, _ = asyncio.run(env._get_pooled_container("img"))
    assert c3 is new
    assert c1.removed
    assert c1.id not in env._CONTAINER_DIRS


def test_backoff_and_consecutive_failures(monkeypatch, tmp_path):
    _reset(monkeypatch)
    monkeypatch.setattr(env, "_CREATE_BACKOFF_BASE", 1.0)
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)
    monkeypatch.setattr(env, "_POOL_METRICS_FILE", tmp_path / "pool.json")

    delays = []
    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(env.asyncio, "sleep", fake_sleep)

    class FailingContainers:
        def __init__(self):
            self.count = 0
        def run(self, *a, **k):
            self.count += 1
            if self.count <= 3:
                raise RuntimeError("fail")
            return DummyContainer(self.count)

    dummy = DummyClient()
    dummy.containers = FailingContainers()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            asyncio.run(env._get_pooled_container("img"))

    assert env._CONSECUTIVE_CREATE_FAILURES["img"] == 3
    assert env._CREATE_FAILURES["img"] == 3

    c, _ = asyncio.run(env._get_pooled_container("img"))
    assert isinstance(c, DummyContainer)
    assert env._CONSECUTIVE_CREATE_FAILURES["img"] == 0
    assert delays and delays[0] == 8.0


def test_create_pool_container_respects_limit(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setattr(env, "_MAX_CONTAINER_COUNT", 1)
    monkeypatch.setattr(env, "_read_active_containers", lambda: ["x"])
    env._ACTIVE_CONTAINER_LIMIT_REACHED = 0

    with pytest.raises(RuntimeError):
        asyncio.run(env._create_pool_container("img"))

    assert env._ACTIVE_CONTAINER_LIMIT_REACHED == 1


def test_warmup_respects_container_limit(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    env._CONTAINER_POOLS.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda *_, **__: None)
    monkeypatch.setattr(env, "_MAX_CONTAINER_COUNT", 1)
    monkeypatch.setattr(env, "_read_active_containers", lambda: ["x"])
    env._ACTIVE_CONTAINER_LIMIT_REACHED = 0
    called = []
    monkeypatch.setattr(env, "_schedule_coroutine", lambda c: called.append(True))

    env._ensure_pool_size_async("img")

    assert not called
    assert env._ACTIVE_CONTAINER_LIMIT_REACHED == 1
