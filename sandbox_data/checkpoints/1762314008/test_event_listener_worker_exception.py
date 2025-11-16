import time
import types
import sandbox_runner.environment as env


def test_worker_exception_triggers_restart(monkeypatch):
    """ensure_cleanup_worker should spawn new thread after worker failure"""

    calls = []
    original_start = env.start_container_event_listener

    def wrapper():
        calls.append(True)
        original_start()

    monkeypatch.setattr(env, "start_container_event_listener", wrapper)
    def fake_schedule(coro):
        coro.close()
        return types.SimpleNamespace()

    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    env._CLEANUP_TASK = None
    env._REAPER_TASK = None

    class BadAPIClient:
        def __init__(self):
            raise RuntimeError("boom")

    failing_docker = types.SimpleNamespace(APIClient=BadAPIClient)
    monkeypatch.setattr(env, "docker", failing_docker)

    env.start_container_event_listener()
    time.sleep(0.05)
    assert env._EVENT_THREAD is not None
    assert not env._EVENT_THREAD.is_alive()

    class DummyAPIClient:
        def __init__(self):
            pass
        def events(self, *a, **k):
            yield {}
        def close(self):
            pass

    working_docker = types.SimpleNamespace(APIClient=DummyAPIClient)
    monkeypatch.setattr(env, "docker", working_docker)

    env.ensure_cleanup_worker()
    time.sleep(0.05)

    assert len(calls) == 2
    assert env._EVENT_THREAD is not None and env._EVENT_THREAD.is_alive()
    env.stop_container_event_listener()
