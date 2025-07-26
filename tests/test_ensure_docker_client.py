import types
import sandbox_runner.environment as env


def test_reconnect_when_ping_fails(monkeypatch):
    class DummyErr(Exception):
        pass

    class BadClient:
        def ping(self):
            raise DummyErr('boom')

    class GoodClient:
        def __init__(self):
            self.ping_called = 0
        def ping(self):
            self.ping_called += 1

    good = GoodClient()

    docker_mod = types.SimpleNamespace(from_env=lambda: good)
    monkeypatch.setattr(env, 'docker', docker_mod)
    monkeypatch.setattr(env, 'DockerException', DummyErr)
    env._DOCKER_CLIENT = BadClient()

    env.ensure_docker_client()
    assert env._DOCKER_CLIENT is good
    assert good.ping_called == 1


def test_no_reconnect_when_ping_ok(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.ping_called = 0
        def ping(self):
            self.ping_called += 1

    client = DummyClient()
    env._DOCKER_CLIENT = client
    monkeypatch.setattr(env, 'docker', types.SimpleNamespace(from_env=lambda: None))

    env.ensure_docker_client()
    assert env._DOCKER_CLIENT is client
    assert client.ping_called == 1
