import asyncio
import importlib
import types
import sys
import sandbox_runner.environment as env


def _stub_docker(cont_map):
    class DummyContainer:
        def __init__(self, label):
            self.id = f"{label}_{len(cont_map.get(label, []))}"
            self.removed = False

        def wait(self, *a, **k):
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
            label = next(iter(kwargs.get("labels", {})))
            c = DummyContainer(label)
            cont_map.setdefault(label, []).append(c)
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


def _setup(monkeypatch, label, tmp_path):
    monkeypatch.setenv("SANDBOX_POOL_LABEL", label)
    importlib.reload(env)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(env, "_record_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_read_active_containers", lambda: [])
    monkeypatch.setattr(env, "_write_active_containers", lambda ids: None)
    monkeypatch.setattr(env, "_read_active_overlays", lambda: [])
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=False: 0)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)


def test_pool_label_cleanup(monkeypatch, tmp_path):
    containers = {}
    _stub_docker(containers)

    _setup(monkeypatch, "label1", tmp_path)
    asyncio.run(env._execute_in_container("print('a')", {"CPU_LIMIT": "1"}))
    c1 = containers["label1"][0]

    _setup(monkeypatch, "label2", tmp_path)
    asyncio.run(env._execute_in_container("print('b')", {"CPU_LIMIT": "1"}))
    c2 = containers["label2"][0]

    assert not c1.removed and not c2.removed

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:4] == ["docker", "ps", "-aq", "--filter"]:
            label = cmd[4].split("=")[1]
            ids = [c.id for c in containers.get(label, []) if not c.removed]
            return types.SimpleNamespace(returncode=0, stdout="\n".join(ids))
        if cmd[:3] == ["docker", "rm", "-f"]:
            cid = cmd[3]
            for lst in containers.values():
                for c in lst:
                    if c.id == cid:
                        c.removed = True
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    monkeypatch.setattr(env, "_POOL_LABEL", "label1")
    env.purge_leftovers()
    assert c1.removed
    assert not c2.removed

    monkeypatch.setattr(env, "_POOL_LABEL", "label2")
    env.purge_leftovers()
    assert c2.removed

    assert all(c.removed for lst in containers.values() for c in lst)
