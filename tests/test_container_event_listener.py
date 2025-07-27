import time
import types
import sandbox_runner.environment as env

class DummyContainer:
    def __init__(self, cid):
        self.id = cid
        self.stopped = False
        self.removed = False
        self.status = "running"
        self.attrs = {}
    def reload(self):
        pass
    def stop(self, timeout=0):
        self.stopped = True
    def remove(self, force=True):
        self.removed = True

class DummyContainers:
    def __init__(self):
        self.items = []
    def get(self, cid):
        for c in self.items:
            if c.id == cid:
                return c
        raise KeyError(cid)

class DummyClient:
    def __init__(self):
        self.containers = DummyContainers()

class DummyAPIClient:
    def __init__(self):
        pass
    def events(self, *a, **k):
        yield {"Type": "container", "Action": "die", "id": "c1"}
    def close(self):
        pass


def test_exited_container_removed(monkeypatch, tmp_path):
    client = DummyClient()
    container = DummyContainer("c1")
    client.containers.items.append(container)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", client)
    monkeypatch.setattr(env, "docker", types.SimpleNamespace(APIClient=DummyAPIClient))
    monkeypatch.setattr(env, "_schedule_coroutine", lambda c: types.SimpleNamespace())

    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._CLEANUP_METRICS.clear()

    env._CONTAINER_POOLS["img"] = [container]
    env._CONTAINER_DIRS[container.id] = str(tmp_path)
    env._CONTAINER_LAST_USED[container.id] = env.time.time()

    env.ensure_cleanup_worker()
    time.sleep(0.05)
    env.stop_container_event_listener()

    assert container.stopped and container.removed
    assert container.id not in env._CONTAINER_DIRS
    assert env._CLEANUP_METRICS.get("event", 0) >= 1


def test_event_listener_api_error(monkeypatch):
    class DummyDockerException(Exception):
        pass

    class BadAPIClient:
        def __init__(self):
            raise DummyDockerException("fail")

    monkeypatch.setattr(env, "DockerException", DummyDockerException, raising=False)
    monkeypatch.setattr(env, "docker", types.SimpleNamespace(APIClient=BadAPIClient))
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    env.start_container_event_listener()
    time.sleep(0.05)
    assert env._EVENT_THREAD is None
