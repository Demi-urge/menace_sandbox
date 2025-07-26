import time
import types
import sandbox_runner.environment as env


class DummyTask:
    def __init__(self):
        self._cancelled = False
        self._done = False

    def cancel(self):
        self._cancelled = True
        self._done = True

    def done(self):
        return self._done

    def cancelled(self):
        return self._cancelled

    def exception(self):
        return None


def test_watchdog_restarts_cancelled_tasks(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    def fake_schedule(coro):
        coro.close()
        return DummyTask()

    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)

    monkeypatch.setattr(
        env,
        "start_container_event_listener",
        lambda: setattr(
            env, "_EVENT_THREAD", types.SimpleNamespace(is_alive=lambda: True)
        ),
    )
    monkeypatch.setattr(env, "stop_container_event_listener", lambda: setattr(env, "_EVENT_THREAD", None))

    env._CLEANUP_TASK = DummyTask()
    env._REAPER_TASK = DummyTask()

    env.schedule_cleanup_check(interval=0.05)

    try:
        time.sleep(0.06)
        orig_cleanup = env._CLEANUP_TASK
        orig_reaper = env._REAPER_TASK

        orig_cleanup.cancel()
        orig_reaper.cancel()

        time.sleep(0.1)

        assert env._CLEANUP_TASK is not orig_cleanup
        assert env._REAPER_TASK is not orig_reaper
        assert not env._CLEANUP_TASK.cancelled()
        assert not env._REAPER_TASK.cancelled()
    finally:
        env.cancel_cleanup_check()
        env._CLEANUP_TASK = None
        env._REAPER_TASK = None
        env.stop_container_event_listener()


def test_watchdog_restarts_stalled_workers(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    created = []

    def fake_schedule(coro):
        coro.close()
        t = DummyTask()
        created.append(t)
        return t

    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)
    monkeypatch.setattr(
        env,
        "start_container_event_listener",
        lambda: setattr(env, "_EVENT_THREAD", types.SimpleNamespace(is_alive=lambda: True)),
    )

    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._CLEANUP_TASK = DummyTask()
    env._REAPER_TASK = DummyTask()
    env._LAST_CLEANUP_TS = time.monotonic() - 3 * env._POOL_CLEANUP_INTERVAL
    env._LAST_REAPER_TS = env._LAST_CLEANUP_TS
    env._WATCHDOG_METRICS.clear()

    env.watchdog_check()

    assert env._CLEANUP_TASK in created
    assert env._REAPER_TASK in created
    assert env._WATCHDOG_METRICS["cleanup"] >= 1
    assert env._WATCHDOG_METRICS["reaper"] >= 1
    env._CLEANUP_TASK = None
    env._REAPER_TASK = None
    env.stop_container_event_listener()
