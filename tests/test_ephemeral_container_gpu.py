import asyncio
import sys
import types

import sandbox_runner.environment as env


def _stub_docker(holder):
    class DummyContainer:
        def __init__(self):
            self.id = "dummy"

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
            pass

        def stop(self, timeout=0):
            pass

    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            holder.append(kwargs)
            return DummyContainer()

    class DummyClient:
        def __init__(self):
            self.containers = DummyContainers()

    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: DummyClient()

    types_mod = types.ModuleType("docker.types")

    class DeviceRequest:
        def __init__(self, *a, **kw):
            self.args = a
            self.kwargs = kw

    types_mod.DeviceRequest = DeviceRequest
    dummy.types = types_mod

    errors_mod = types.ModuleType("docker.errors")

    class DummyErr(Exception):
        pass

    errors_mod.DockerException = DummyErr
    errors_mod.APIError = DummyErr
    dummy.errors = errors_mod

    sys.modules["docker"] = dummy
    sys.modules["docker.types"] = types_mod
    sys.modules["docker.errors"] = errors_mod


def test_ephemeral_container_gpu(monkeypatch):
    calls = []
    _stub_docker(calls)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._WARMUP_TASKS.clear()

    res = asyncio.run(
        env._execute_in_container("print('hi')", {"GPU_LIMIT": "1"}, network_disabled=False)
    )

    assert res["exit_code"] == 0.0
    assert calls and "device_requests" in calls[0]
    assert calls[0].get("network_disabled") is False

