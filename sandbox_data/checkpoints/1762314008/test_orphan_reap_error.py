import types
import sandbox_runner.environment as env

class DummyDockerException(Exception):
    pass

class BadList:
    def list(self, *a, **k):
        raise DummyDockerException("fail")

class DummyClient:
    def __init__(self):
        self.containers = BadList()

def test_reap_orphan_handles_error(monkeypatch):
    monkeypatch.setattr(env, "DockerException", DummyDockerException, raising=False)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", DummyClient())
    removed = env._reap_orphan_containers()
    assert removed == 0

