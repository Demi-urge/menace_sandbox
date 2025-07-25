import types
import sandbox_runner.environment as env


class DummyTask:
    def __init__(self, done=False):
        self._done = done

    def done(self):
        return self._done

    def cancelled(self):
        return False

    def exception(self):
        return None


def test_restart_dead_event_listener(monkeypatch):
    """ensure_cleanup_worker should restart a stopped event listener"""

    called = []
    monkeypatch.setattr(env, "start_container_event_listener", lambda: called.append(True))
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: False)
    env._CLEANUP_TASK = DummyTask(done=False)
    env._REAPER_TASK = DummyTask(done=False)

    env.ensure_cleanup_worker()

    assert called
