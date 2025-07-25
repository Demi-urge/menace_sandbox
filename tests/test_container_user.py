import asyncio
import contextlib
import types
import sys
import sandbox_runner.environment as env


class DummyContainer:
    def __init__(self):
        self.id = "dummy"
        self.removed = False

    def wait(self, *a, **k):
        return {"StatusCode": 0}

    def logs(self, stdout=True, stderr=False):
        return b""

    def stats(self, stream=False):
        return {
            "blkio_stats": {"io_service_bytes_recursive": []},
            "cpu_stats": {"cpu_usage": {"total_usage": 1}},
            "memory_stats": {"max_usage": 1},
            "networks": {},
        }

    def remove(self):
        self.removed = True

    def stop(self, timeout=0):
        pass


class DummyContainers:
    def __init__(self, holder):
        self.holder = holder

    def run(self, image, cmd, **kwargs):
        self.holder.append(kwargs)
        return DummyContainer()


class DummyClient:
    def __init__(self, holder):
        self.containers = DummyContainers(holder)


def _stub_docker(holder):
    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: DummyClient(holder)
    dummy.types = types
    err_mod = types.ModuleType("docker.errors")
    class DummyErr(Exception):
        pass
    err_mod.DockerException = DummyErr
    err_mod.APIError = DummyErr
    dummy.errors = err_mod
    sys.modules["docker"] = dummy
    sys.modules["docker.errors"] = err_mod


def test_pool_container_user(monkeypatch, tmp_path):
    kwargs_seen = []
    _stub_docker(kwargs_seen)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient(kwargs_seen))
    monkeypatch.setenv("SANDBOX_CONTAINER_USER", "1000")
    monkeypatch.setattr(env, "_CONTAINER_USER", "1000")
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
    assert kwargs_seen and kwargs_seen[0].get("user") == "1000"


def test_ephemeral_container_user(monkeypatch):
    kwargs_seen = []
    _stub_docker(kwargs_seen)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._WARMUP_TASKS.clear()
    monkeypatch.setenv("SANDBOX_CONTAINER_USER", "1001")
    res = asyncio.run(env._execute_in_container("print('hi')", {"CPU_LIMIT": "1"}))
    assert res["exit_code"] == 0.0
    assert kwargs_seen and kwargs_seen[0].get("user") == "1001"

