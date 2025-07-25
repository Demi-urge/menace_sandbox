import types
import sandbox_runner.environment as env


class DummyTask:
    def __init__(self, done=True, exc=None, cancelled=False):
        self._done = done
        self._exc = exc
        self._cancelled = cancelled

    def done(self):
        return self._done

    def cancelled(self):
        return self._cancelled

    def exception(self):
        return self._exc


def test_scheduler_restarts_cancelled(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    tasks = [object(), object()]

    def fake_schedule(coro):
        coro.close()
        return tasks.pop(0)

    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)

    orig_cleanup = DummyTask(done=True, cancelled=True)
    orig_reaper = DummyTask(done=True, cancelled=True)
    env._CLEANUP_TASK = orig_cleanup
    env._REAPER_TASK = orig_reaper

    class FakeTimer:
        def __init__(self, interval, func):
            self.func = func
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(env.threading, "Timer", FakeTimer)

    env.schedule_cleanup_check(interval=1)
    timer = env._WORKER_CHECK_TIMER
    timer.func()

    assert env._CLEANUP_TASK is not orig_cleanup
    assert env._REAPER_TASK is not orig_reaper
    assert env._WORKER_CHECK_TIMER is not timer

    env.cancel_cleanup_check()


def test_scheduler_restarts_on_exception(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    tasks = [object(), object()]

    def fake_schedule(coro):
        coro.close()
        return tasks.pop(0)

    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)

    orig_cleanup = DummyTask(done=True, exc=RuntimeError("boom"))
    orig_reaper = DummyTask(done=True, exc=RuntimeError("boom"))
    env._CLEANUP_TASK = orig_cleanup
    env._REAPER_TASK = orig_reaper

    class FakeTimer:
        def __init__(self, interval, func):
            self.func = func
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(env.threading, "Timer", FakeTimer)

    env.schedule_cleanup_check(interval=1)
    timer = env._WORKER_CHECK_TIMER
    timer.func()

    assert env._CLEANUP_TASK is not orig_cleanup
    assert env._REAPER_TASK is not orig_reaper
    assert env._WORKER_CHECK_TIMER is not timer

    env.cancel_cleanup_check()


def test_watchdog_restarts_dead_event_thread(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    called = []
    def fake_start_listener():
        called.append(True)
        env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)

    monkeypatch.setattr(env, "start_container_event_listener", fake_start_listener)

    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: False)
    env._CLEANUP_TASK = DummyTask(done=False)
    env._REAPER_TASK = DummyTask(done=False)
    env._WATCHDOG_METRICS.clear()

    class FakeTimer:
        def __init__(self, interval, func):
            self.func = func
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(env.threading, "Timer", FakeTimer)

    env.schedule_cleanup_check(interval=1)
    timer = env._WORKER_CHECK_TIMER
    timer.func()

    assert called
    assert env._WATCHDOG_METRICS.get("event", 0) >= 1
    assert env._WORKER_CHECK_TIMER is not timer

    env.cancel_cleanup_check()
