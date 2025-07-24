import asyncio
import contextlib
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


class FailContainer(DummyContainer):
    def stop(self, timeout=0):
        raise RuntimeError("stop fail")

    def remove(self, force=True):
        raise RuntimeError("remove fail")


class BadContainer(DummyContainer):
    def __init__(self, cid, fail_reload=False):
        super().__init__(cid)
        self.fail_reload = fail_reload

    def reload(self):
        if self.fail_reload:
            raise RuntimeError("boom")
        self.status = "exited"


def test_idle_container_cleanup(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 0.1)
    monkeypatch.setattr(env, "_CONTAINER_POOL_SIZE", 1)
    times = [0.0]
    monkeypatch.setattr(env.time, "time", lambda: times[0])
    c = DummyContainer("seed")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = "dir"
    env._CONTAINER_LAST_USED[c.id] = 0.0
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    assert len(env._CONTAINER_POOLS["img"]) == env._CONTAINER_POOL_SIZE
    c, _ = asyncio.run(env._get_pooled_container("img"))
    env._release_container("img", c)
    if "img" in env._WARMUP_TASKS:
        asyncio.run(env._WARMUP_TASKS["img"])
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
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_CONTAINER_POOL_SIZE", 1)
    pool = [DummyContainer(i) for i in range(env._CONTAINER_POOL_SIZE)]
    env._CONTAINER_POOLS["img"] = pool
    now = 0.0
    monkeypatch.setattr(env.time, "time", lambda: now)
    for c in pool:
        env._CONTAINER_LAST_USED[c.id] = now
    env._ensure_pool_size_async("img")
    assert "img" not in env._WARMUP_TASKS


def test_cleanup_logs_errors(monkeypatch, tmp_path, caplog):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(env.time, "time", lambda: 1.0)
    c = FailContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = 0.0
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    caplog.set_level("ERROR")
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 1 and replaced == 0
    assert "failed to stop container" in caplog.text
    assert "failed to remove container" in caplog.text


def test_cleanup_worker_logs_stats(monkeypatch, caplog):
    calls = []

    async def run_worker():
        task = asyncio.create_task(env._cleanup_worker())
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda: calls.append(1) or (2, 1))
    caplog.set_level("INFO")
    asyncio.run(run_worker())
    assert calls
    assert "cleaned 2 idle containers" in caplog.text
    assert "replaced 1 unhealthy containers" in caplog.text


def test_cleanup_worker_purges_vms(monkeypatch):
    calls = []

    async def run_worker():
        task = asyncio.create_task(env._cleanup_worker())
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda: (0, 0))

    def fake_purge(*, record_runtime=False):
        calls.append(record_runtime)
        env._RUNTIME_VMS_REMOVED += 1
        env._STALE_VMS_REMOVED += 1
        return 1

    monkeypatch.setattr(env, "_purge_stale_vms", fake_purge)
    env._RUNTIME_VMS_REMOVED = 0
    env._STALE_VMS_REMOVED = 0
    asyncio.run(run_worker())
    assert calls and calls[0] is True
    assert env._RUNTIME_VMS_REMOVED >= 1


def test_rmtree_failure_in_idle_cleanup(monkeypatch, tmp_path, caplog):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setattr(env, "_CONTAINER_IDLE_TIMEOUT", 0.0)
    monkeypatch.setattr(env.time, "time", lambda: 1.0)
    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    td = tmp_path / "dir"
    td.mkdir()
    env._CONTAINER_DIRS[c.id] = str(td)
    env._CONTAINER_LAST_USED[c.id] = 0.0

    orig_rmtree = env.shutil.rmtree

    def failing(path):
        orig_rmtree(path)
        raise RuntimeError("boom")

    monkeypatch.setattr(env.shutil, "rmtree", failing)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    caplog.set_level("ERROR")
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 1 and replaced == 0
    assert not td.exists()
    assert "temporary directory removal failed" in caplog.text


def test_rmtree_failure_in_pool_cleanup(monkeypatch, tmp_path, caplog):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None
    c = DummyContainer("x")
    env._CONTAINER_POOLS["img"] = [c]
    td = tmp_path / "dir"
    td.mkdir()
    env._CONTAINER_DIRS[c.id] = str(td)
    env._CONTAINER_LAST_USED[c.id] = 0.0

    orig_rmtree = env.shutil.rmtree

    def failing(path):
        orig_rmtree(path)
        raise RuntimeError("boom")

    monkeypatch.setattr(env.shutil, "rmtree", failing)
    caplog.set_level("ERROR")
    env._cleanup_pools()
    assert not td.exists()
    assert "temporary directory removal failed" in caplog.text


def test_unhealthy_container_replaced_on_get(monkeypatch, tmp_path):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    bad = BadContainer("x")
    env._CONTAINER_POOLS["img"] = [bad]
    env._CONTAINER_DIRS[bad.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[bad.id] = env.time.time()

    new = DummyContainer("new")

    async def create(image):
        return new, str(tmp_path / "n")

    monkeypatch.setattr(env, "_create_pool_container", create)
    c, td = asyncio.run(env._get_pooled_container("img"))
    assert c is new
    assert bad.removed
    assert bad.id not in env._CONTAINER_DIRS


def test_unhealthy_container_purged_in_cleanup(monkeypatch, tmp_path):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    monkeypatch.setattr(env, "_ensure_pool_size_async", lambda img: None)
    bad = BadContainer("x", fail_reload=True)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_POOLS["img"] = [bad]
    env._CONTAINER_DIRS[bad.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[bad.id] = env.time.time()
    cleaned, replaced = env._cleanup_idle_containers()
    assert cleaned == 0 and replaced == 1
    assert bad.removed
    assert not env._CONTAINER_POOLS["img"]
