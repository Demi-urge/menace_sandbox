import asyncio
import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid):
        self.id = f"c{cid}"
        self.status = "running"
        self.removed = False
        self.stopped = False
        self.attrs = {"State": {"Health": {"Status": "healthy"}}}

    def reload(self):
        pass

    def stop(self, timeout=0):
        self.stopped = True

    def remove(self, force=True):
        self.removed = True

class DummyClient:
    def __init__(self):
        self.containers = type("C", (), {"run": lambda *a, **k: DummyContainer("new")})()


def _setup(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_CREATED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_METRICS.clear()


def test_disk_usage_cleanup(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 999)
    monkeypatch.setattr(env, "_CONTAINER_MAX_LIFETIME", 999)
    monkeypatch.setattr(env, "_CONTAINER_DISK_LIMIT", 1)
    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = env.time.time()
    env._CONTAINER_CREATED[c.id] = env.time.time()
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    monkeypatch.setattr(env, "_get_dir_usage", lambda p: 10)
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 0 and replaced == 1
    assert env._CLEANUP_METRICS["disk"] == 1
    metrics = asyncio.run(env.collect_metrics_async(0.0, 0.0, None))
    assert metrics.get("cleanup_disk") == 1.0


def test_lifetime_cleanup(monkeypatch, tmp_path):
    _setup(monkeypatch)
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 999)
    monkeypatch.setattr(env, "_CONTAINER_MAX_LIFETIME", 0.1)
    monkeypatch.setattr(env, "_CONTAINER_DISK_LIMIT", 0)
    times = [0.0]
    monkeypatch.setattr(env.time, "time", lambda: times[0])
    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = 0.0
    env._CONTAINER_CREATED[c.id] = 0.0
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    times[0] = 0.2
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 0 and replaced == 1
    assert env._CLEANUP_METRICS["lifetime"] == 1


def test_get_dir_usage_error_logged(monkeypatch, tmp_path, caplog):
    path = tmp_path
    (path / "f.txt").write_text("data")

    def raise_error(_p):
        raise OSError("fail")

    monkeypatch.setattr(env.os.path, "getsize", raise_error)
    caplog.set_level("WARNING")
    assert env._get_dir_usage(str(path)) == 0
    assert f"size check failed for {str(path)}" in caplog.text

