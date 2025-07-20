import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid):
        self.id = f"c{cid}"
        self.status = "running"
        self.stopped = False
        self.removed = False
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


def test_idle_container_cleanup(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_THREADS.clear()
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 0.1)
    monkeypatch.setattr(env, "_CONTAINER_POOL_SIZE", 1)
    times = [0.0]
    monkeypatch.setattr(env.time, "time", lambda: times[0])
    env._ensure_pool_size_async("img")
    th = env._WARMUP_THREADS["img"]
    th.join()
    assert len(env._CONTAINER_POOLS["img"]) == env._CONTAINER_POOL_SIZE
    c, _ = env._get_pooled_container("img")
    env._release_container("img", c)
    if "img" in env._WARMUP_THREADS:
        env._WARMUP_THREADS["img"].join()
    times[0] = 0.2
    env._cleanup_idle_containers()
    assert not env._CONTAINER_POOLS["img"]
    assert c.stopped and c.removed


def test_pool_not_expanded_when_full(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_THREADS.clear()
    monkeypatch.setattr(env, "_CONTAINER_POOL_SIZE", 1)
    pool = [DummyContainer(i) for i in range(env._CONTAINER_POOL_SIZE)]
    env._CONTAINER_POOLS["img"] = pool
    now = 0.0
    monkeypatch.setattr(env.time, "time", lambda: now)
    for c in pool:
        env._CONTAINER_LAST_USED[c.id] = now
    env._ensure_pool_size_async("img")
    assert "img" not in env._WARMUP_THREADS
