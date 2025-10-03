import asyncio
import contextlib
import types
import sys
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
        self.items = []
    def run(self, *a, **k):
        self.count += 1
        c = DummyContainer(self.count)
        self.items.append(c)
        return c
    def list(self, *a, **k):
        return list(self.items)
    def get(self, cid):
        for c in self.items:
            if c.id == cid:
                return c
        raise KeyError(cid)

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


def test_cleanup_worker_calls_retry(monkeypatch):
    called = []

    async def run_worker():
        task = asyncio.create_task(env._cleanup_worker())
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(
        env, "retry_failed_cleanup", lambda progress=None: called.append(True) or (0, 0)
    )
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda: (0, 0))
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)

    asyncio.run(run_worker())

    assert called

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
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)
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

def test_orphan_container_reaped(monkeypatch, tmp_path):
    dummy = DummyClient()
    monkeypatch.setattr(env, "_DOCKER_CLIENT", dummy)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CONTAINER_CREATED.clear()
    env._WARMUP_TASKS.clear()
    c = DummyContainer("x")
    dummy.containers.items.append(c)
    env._CONTAINER_DIRS[c.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[c.id] = env.time.time()
    env._CONTAINER_CREATED[c.id] = env.time.time()
    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    async def run_worker():
        task = asyncio.create_task(env._reaper_worker())
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    asyncio.run(run_worker())
    assert c.stopped and c.removed
    assert c.id not in env._CONTAINER_DIRS

def test_stop_and_remove_detects_remaining_container(monkeypatch, caplog):
    c = DummyContainer("x")
    env._CLEANUP_FAILURES = 0
    called = []
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: called.append(cid))
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout=f"{c.id}\n")
    monkeypatch.setattr(env.subprocess, "run", fake_run)
    caplog.set_level("ERROR")
    env._stop_and_remove(c)
    assert env._CLEANUP_FAILURES == 1
    assert not called
    assert f"container {c.id} still exists" in caplog.text

def test_stop_and_remove_removes_active_on_success(monkeypatch):
    c = DummyContainer("y")
    env._CLEANUP_FAILURES = 0
    called = []
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: called.append(cid))
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""))
    env._stop_and_remove(c)
    assert called == [c.id]
    assert env._CLEANUP_FAILURES == 0


def test_stop_and_remove_uses_cli_fallback(monkeypatch):
    class RemoveFail(DummyContainer):
        def remove(self, force=True):
            raise RuntimeError("boom")

    c = RemoveFail("z")
    cmds = []
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: None)

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return types.SimpleNamespace(returncode=0, stdout=f"{c.id}\n")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    env._stop_and_remove(c)
    assert any(cmd[:3] == ["docker", "rm", "-f"] for cmd in cmds)


def test_stop_and_remove_force_kills(monkeypatch):
    c = DummyContainer("f")
    env._FORCE_KILLS = 0
    called = []
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: called.append(cid))

    ps_calls = 0
    def fake_run(cmd, **kw):
        nonlocal ps_calls
        if cmd[:2] == ["docker", "ps"]:
            ps_calls += 1
            if ps_calls < 3:
                return types.SimpleNamespace(returncode=0, stdout=f"{c.id}\n")
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    env._stop_and_remove(c)
    assert env._FORCE_KILLS == 1
    assert called == [c.id]


def test_pool_container_disk_limit(monkeypatch, tmp_path):
    """Verify storage limit option passed to Docker."""
    kwargs_seen = {}

    class CapContainers(DummyContainers):
        def run(self, *a, **k):
            kwargs_seen.update(k)
            return super().run(*a, **k)

    class CapClient:
        def __init__(self):
            self.containers = CapContainers()

    monkeypatch.setenv("SANDBOX_CONTAINER_DISK_LIMIT", "42")
    monkeypatch.setattr(env, "_CONTAINER_DISK_LIMIT", env._parse_size("42"))
    monkeypatch.setattr(env, "_DOCKER_CLIENT", CapClient())
    monkeypatch.setattr(env, "_record_active_container", lambda cid: None)

    @contextlib.asynccontextmanager
    async def fake_lock():
        yield

    monkeypatch.setattr(env, "pool_lock", fake_lock)

    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()

    asyncio.run(env._create_pool_container("img"))
    assert kwargs_seen.get("storage_opt") == {"size": "42"}


def test_cleanup_worker_records_duration(monkeypatch):
    stub = types.ModuleType("metrics_exporter")
    class DummyGauge:
        def __init__(self):
            self.values = {}
        def labels(self, worker):
            def set_val(v):
                self.values[worker] = v
            return types.SimpleNamespace(set=set_val)
    stub.cleanup_duration_gauge = DummyGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda: (0, 0))
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)

    async def run_worker():
        task = asyncio.create_task(env._cleanup_worker())
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_worker())

    assert stub.cleanup_duration_gauge.values.get("cleanup", 0) > 0
    assert env._CLEANUP_DURATIONS["cleanup"] > 0
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["cleanup_duration_seconds_cleanup"] > 0


def test_reaper_worker_records_duration(monkeypatch):
    stub = types.ModuleType("metrics_exporter")
    class DummyGauge:
        def __init__(self):
            self.values = {}
        def labels(self, worker):
            def set_val(v):
                self.values[worker] = v
            return types.SimpleNamespace(set=set_val)
    stub.cleanup_duration_gauge = DummyGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(env, "_reap_orphan_containers", lambda: 0)

    async def run_worker():
        task = asyncio.create_task(env._reaper_worker())
        await asyncio.sleep(0.03)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_worker())

    assert stub.cleanup_duration_gauge.values.get("reaper", 0) > 0
    assert env._CLEANUP_DURATIONS["reaper"] > 0
    metrics = env.collect_metrics(0.0, 0.0, None)
    assert metrics["cleanup_duration_seconds_reaper"] > 0


def test_watchdog_restarts_failed_workers(monkeypatch, caplog):
    """Ensure watchdog restarts workers that exited with an error."""

    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    class DummyTask:
        def __init__(self, exc=None):
            self._exc = exc

        def cancel(self):
            pass

        def done(self):
            return True

        def cancelled(self):
            return False

        def exception(self):
            return self._exc

    created = []

    def fake_schedule(coro):
        coro.close()
        t = DummyTask()
        created.append(t)
        return t

    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)

    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._CLEANUP_TASK = DummyTask(RuntimeError("boom"))
    env._REAPER_TASK = DummyTask(RuntimeError("boom"))
    env._WATCHDOG_METRICS.clear()
    caplog.set_level("WARNING")

    env.watchdog_check()

    assert env._CLEANUP_TASK in created
    assert env._REAPER_TASK in created
    assert "cleanup worker restarted by watchdog" in caplog.text
    assert "reaper worker restarted by watchdog" in caplog.text
