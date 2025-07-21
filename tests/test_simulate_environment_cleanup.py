import types
import asyncio
import sys
import sandbox_runner.environment as env


def _stub_docker(calls):
    class DummyExec:
        def __init__(self, code=0):
            self.exit_code = code

    class DummyContainer:
        id = "dummy"
        def wait(self):
            return {"StatusCode": 0}
        def exec_run(self, *a, **k):
            return DummyExec()
        def stats(self, stream=False):
            return {
                "blkio_stats": {"io_service_bytes_recursive": []},
                "cpu_stats": {"cpu_usage": {"total_usage": 1}},
                "memory_stats": {"max_usage": 1},
            }
        def remove(self, force=True):
            calls.append("removed")
        def stop(self, timeout=0):
            calls.append("stopped")

    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            calls.append("run")
            return DummyContainer()

    class DummyClient:
        containers = DummyContainers()

    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: DummyClient()
    dummy.types = types
    sys.modules["docker"] = dummy


def test_simulate_execution_cleanup(monkeypatch):
    calls = []
    _stub_docker(calls)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None

    env.simulate_execution_environment("print('hi')", {"CPU_LIMIT": "1"}, container=True)

    assert "run" in calls and "removed" in calls
    assert not env._CONTAINER_POOLS
