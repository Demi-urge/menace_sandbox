import logging
import os
import sys
import threading
import time
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

sys.modules.pop("sandbox_runner.environment", None)
sys.modules.pop("sandbox_runner", None)
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


def test_stop_listener_terminates_thread(monkeypatch):
    stop_event = threading.Event()

    def worker():
        stop_event.wait()

    thread = threading.Thread(target=worker)
    thread.start()
    env._EVENT_THREAD = thread
    env._EVENT_STOP = stop_event

    env.stop_container_event_listener()

    assert env._EVENT_THREAD is None
    assert not thread.is_alive()


def test_stop_listener_retries_on_error(monkeypatch, caplog):
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            time.sleep(0.01)
        # give time so the thread is still alive after first join failure
        time.sleep(0.2)

    thread = threading.Thread(target=worker)
    thread.start()
    env._EVENT_THREAD = thread
    env._EVENT_STOP = stop_event

    original_join = thread.join
    calls = {"n": 0}

    def flaky_join(timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original_join(timeout)

    monkeypatch.setattr(thread, "join", flaky_join)

    with caplog.at_level(logging.ERROR):
        env.stop_container_event_listener()

    assert calls["n"] >= 2
    assert env._EVENT_THREAD is None
    assert not thread.is_alive()
