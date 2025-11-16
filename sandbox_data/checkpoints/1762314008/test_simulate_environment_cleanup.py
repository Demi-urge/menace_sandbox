import types
import asyncio
import subprocess
import sys
import sandbox_runner.environment as env
import pytest


@pytest.fixture(autouse=True)
def _no_event_listener(monkeypatch):
    monkeypatch.setattr(env, "start_container_event_listener", lambda: None)
    monkeypatch.setattr(env, "stop_container_event_listener", lambda: None)


def _stub_docker(calls):
    class DummyExec:
        def __init__(self, code=0):
            self.exit_code = code

    class DummyContainer:
        id = "dummy"
        def wait(self, *a, **k):
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
    class DummyErr(Exception):
        pass
    errors_mod = types.ModuleType("docker.errors")
    errors_mod.DockerException = DummyErr
    errors_mod.APIError = DummyErr
    dummy.errors = errors_mod
    sys.modules["docker.errors"] = errors_mod
    sys.modules["docker"] = dummy


def _stub_docker_timeout(calls):
    class DummyExec:
        def __init__(self, code=0):
            self.exit_code = code

    class DummyContainer:
        id = "dummy"
        def wait(self, *a, **k):
            raise subprocess.TimeoutExpired("wait", 1)
        def exec_run(self, *a, **k):
            raise subprocess.TimeoutExpired("exec", 1)
        def kill(self):
            calls.append("killed")
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
    class DummyErr(Exception):
        pass
    errors_mod = types.ModuleType("docker.errors")
    errors_mod.DockerException = DummyErr
    errors_mod.APIError = DummyErr
    dummy.errors = errors_mod
    sys.modules["docker.errors"] = errors_mod
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

def test_simulate_execution_calls_cleanup_check(monkeypatch):
    called = []
    monkeypatch.setattr(env, "ensure_cleanup_worker", lambda: called.append(True))
    env.simulate_execution_environment("print('hi')", container=False)
    assert called


def test_container_timeout_kills_and_removes(monkeypatch):
    calls = []
    _stub_docker_timeout(calls)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None

    env.simulate_execution_environment("print('hi')", {"TIMEOUT": "0"}, container=True)

    assert "run" in calls and "killed" in calls and "removed" in calls
