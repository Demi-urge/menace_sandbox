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
