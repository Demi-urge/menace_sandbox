import types
import importlib
import sandbox_runner.environment as env

def test_docker_available_handles_exception(monkeypatch):
    class DummyClient:
        def ping(self):
            raise env.DockerException("boom")
    dummy = types.SimpleNamespace(from_env=lambda: DummyClient())
    monkeypatch.setattr(env, "docker", dummy)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(env, "DockerException", RuntimeError)
    assert env._docker_available() is False
