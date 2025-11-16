import sys
import types
import signal
import asyncio
import pytest
import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid="x"):
        self.id = cid
        self.status = "running"
        self.stopped = False
        self.removed = False
    def reload(self):
        pass
    def stop(self, timeout=0):
        self.stopped = True
    def remove(self, force=True):
        self.removed = True


def test_sigterm_triggers_cleanup(monkeypatch, tmp_path):
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CLEANUP_TASK = None
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    c = DummyContainer()
    td = tmp_path / c.id
    td.mkdir()
    env._CONTAINER_POOLS["img"] = [c]
    env._CONTAINER_DIRS[c.id] = str(td)
    env._CONTAINER_LAST_USED[c.id] = env.time.time()

    called = []
    orig = env._cleanup_pools
    def cleanup_wrapper():
        called.append(True)
        orig()
    monkeypatch.setattr(env, "_cleanup_pools", cleanup_wrapper)
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)
    monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    def handler(signum, frame):
        env._cleanup_pools()
        env._await_cleanup_task()
        sys.exit(0)

    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    assert called and c.stopped and c.removed


def test_import_without_loop(monkeypatch):
    if "sandbox_runner.environment" in sys.modules:
        del sys.modules["sandbox_runner.environment"]
    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: None
    monkeypatch.setitem(sys.modules, "docker", dummy)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))
    import importlib
    env2 = importlib.import_module("sandbox_runner.environment")
    assert hasattr(env2, "_cleanup_pools")


def test_stale_containers_removed(monkeypatch):
    env._CONTAINER_POOLS.clear()
    env._CLEANUP_TASK = None
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    removed = []
    def fake_run(cmd, **kw):
        if "ps" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="abc\n")
        if "rm" in cmd:
            removed.append(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)
    env._cleanup_pools()
    assert removed == ["abc"]


def test_cleanup_pool_lock_timeout(monkeypatch):
    class DummyLock:
        def acquire(self, timeout=None):
            raise env.Timeout("locked")

    releases: list[bool] = []
    monkeypatch.setattr(env, "_release_pool_lock", lambda: releases.append(True))
    monkeypatch.setattr(env, "_POOL_FILE_LOCK", DummyLock())

    # should not raise even though the lock cannot be acquired
    env._cleanup_pools()

    # lock was never acquired, so release should not be attempted
    assert not releases
