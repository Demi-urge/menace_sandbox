import os
import sys
import types
import asyncio
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import sys
dummy_jinja = types.ModuleType("jinja2")
dummy_jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", dummy_jinja)
dummy_yaml = types.ModuleType("yaml")
dummy_yaml.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", dummy_yaml)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
import sandbox_runner.environment as env


def _stub_docker(image_holder):
    class DummyExec:
        def __init__(self, code=0):
            self.exit_code = code

    class DummyContainer:
        id = "dummy"
        def exec_run(self, *a, **k):
            return DummyExec(0)

        def wait(self):
            return {"StatusCode": 0}

        def stats(self, stream=False):
            return {
                "blkio_stats": {"io_service_bytes_recursive": []},
                "cpu_stats": {"cpu_usage": {"total_usage": 1}},
                "memory_stats": {"max_usage": 1},
            }

        def remove(self):
            image_holder.append("removed")

        def stop(self, timeout=0):
            pass

    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            image_holder.append(image)
            return DummyContainer()

    class DummyClient:
        containers = DummyContainers()

    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: DummyClient()
    sys.modules["docker"] = dummy


def test_execute_in_container_windows(monkeypatch):
    calls = []
    _stub_docker(calls)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._DOCKER_CLIENT = None
    monkeypatch.setenv("SANDBOX_CONTAINER_IMAGE_WINDOWS", "win-img")
    res = asyncio.run(env._execute_in_container("print('hi')", {"OS_TYPE": "windows"}))
    assert calls[0] == "win-img" and res["exit_code"] == 0.0


def test_execute_in_container_override(monkeypatch):
    calls = []
    _stub_docker(calls)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._DOCKER_CLIENT = None
    res = asyncio.run(
        env._execute_in_container(
            "print('x')", {"OS_TYPE": "macos", "CONTAINER_IMAGE": "custom"}
        )
    )
    assert calls[0] == "custom" and res["exit_code"] == 0.0


def test_execute_in_container_retry(monkeypatch):
    attempts = []

    class FailOnceClient:
        def __init__(self):
            self.called = 0

        class DummyContainer:
            id = "dummy"
            def wait(self):
                return {"StatusCode": 0}

            def stats(self, stream=False):
                return {
                    "blkio_stats": {"io_service_bytes_recursive": []},
                    "cpu_stats": {"cpu_usage": {"total_usage": 1}},
                    "memory_stats": {"max_usage": 1},
                }

            def remove(self):
                pass

        class DummyContainers:
            def run(self, image, cmd, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError("boom")
                return FailOnceClient.DummyContainer()

        containers = DummyContainers()

    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: FailOnceClient()
    dummy.types = types
    sys.modules["docker"] = dummy

    monkeypatch.setattr(env.time, "sleep", lambda s: None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._DOCKER_CLIENT = None
    res = asyncio.run(env._execute_in_container("print('hi')", {}))
    assert len(attempts) >= 2 and res["exit_code"] == 0.0
