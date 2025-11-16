import asyncio
import types
import sys
import sandbox_runner.environment as env


def _stub_docker(containers, labels):
    class DummyContainer:
        def __init__(self):
            self.id = "dummy"
            self.removed = False
        def wait(self, *a, **kw):
            raise RuntimeError("boom")
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
        def run(self, image, cmd, **kwargs):
            labels.append(kwargs.get("labels"))
            c = DummyContainer()
            containers.append(c)
            return c

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

def test_purge_leftovers_removes_ephemeral(monkeypatch):
    containers = []
    labels = []
    monkeypatch.setenv("SANDBOX_POOL_LABEL", "test_label")
    _stub_docker(containers, labels)
    import importlib
    importlib.reload(env)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)

    res = asyncio.run(env._execute_in_container("print('hi')", {"CPU_LIMIT": "1"}))
    assert labels and labels[0] == {env._POOL_LABEL: "1"}
    assert containers and not containers[0].removed

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:4] == ["docker", "ps", "-aq", f"--filter"]:
            return types.SimpleNamespace(stdout=f"{containers[0].id}\n", returncode=0)
        if cmd[:3] == ["docker", "rm", "-f"]:
            if cmd[3] == containers[0].id:
                containers[0].removed = True
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env.purge_leftovers()
    assert containers[0].removed

