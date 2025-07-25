import threading
import time
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

class DummyContainers:
    def __init__(self):
        self.items = []
    def list(self, *a, **k):
        return list(self.items)
    def run(self, *a, **k):
        c = DummyContainer(len(self.items))
        self.items.append(c)
        return c

class DummyClient:
    def __init__(self):
        self.containers = DummyContainers()


def _setup(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_CREATED.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    monkeypatch.setattr(env, "_verify_container", lambda c: True)
    monkeypatch.setattr(env, "_stop_and_remove", lambda c, **k: None)


def test_cleanup_and_reaper_thread_safety(monkeypatch, tmp_path):
    _setup(monkeypatch)
    now = time.time()
    monkeypatch.setattr(env.time, "time", lambda: now)
    client = env._DOCKER_CLIENT

    stop = threading.Event()
    errors: list[Exception] = []

    def worker():
        while not stop.is_set():
            try:
                env._cleanup_idle_containers()
                env._reap_orphan_containers()
            except Exception as exc:  # pragma: no cover - fail fast
                errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()

    for i in range(20):
        c = DummyContainer(f"x{i}")
        client.containers.items.append(c)
        with env._POOL_LOCK:
            env._CONTAINER_POOLS.setdefault("img", []).append(c)
            env._CONTAINER_DIRS[c.id] = str(tmp_path / c.id)
            env._CONTAINER_LAST_USED[c.id] = now
            env._CONTAINER_CREATED[c.id] = now
        time.sleep(0.001)

    stop.set()
    t.join()
    assert not errors
