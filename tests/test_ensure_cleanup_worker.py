import sandbox_runner.environment as env
import pytest


@pytest.fixture(autouse=True)
def _no_event_listener(monkeypatch):
    monkeypatch.setattr(env, "start_container_event_listener", lambda: None)
    monkeypatch.setattr(env, "stop_container_event_listener", lambda: None)

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

def test_restart_when_cancelled(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    new_task = object()
    def fake_schedule(coro):
        coro.close()
        return new_task
    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)
    env._CLEANUP_TASK = DummyTask(done=True, cancelled=True)
    env.ensure_cleanup_worker()
    assert env._CLEANUP_TASK is new_task

def test_no_restart_when_running(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    task = DummyTask(done=False)
    env._CLEANUP_TASK = task
    env.ensure_cleanup_worker()
    assert env._CLEANUP_TASK is task

def test_restart_on_exception(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    new_task = object()
    def fake_schedule(coro):
        coro.close()
        return new_task
    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)
    env._CLEANUP_TASK = DummyTask(done=True, exc=RuntimeError("boom"))
    env.ensure_cleanup_worker()
    assert env._CLEANUP_TASK is new_task


def test_reaper_restart_when_cancelled(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    new_task = object()
    def fake_schedule(coro):
        coro.close()
        return new_task
    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)
    env._CLEANUP_TASK = DummyTask(done=True)
    env._REAPER_TASK = DummyTask(done=True, cancelled=True)
    env.ensure_cleanup_worker()
    assert env._REAPER_TASK is new_task


def test_reaper_no_restart_when_running(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    task = DummyTask(done=False)
    env._CLEANUP_TASK = DummyTask(done=True)
    env._REAPER_TASK = task
    env.ensure_cleanup_worker()
    assert env._REAPER_TASK is task


def test_reaper_restart_on_exception(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    new_task = object()
    def fake_schedule(coro):
        coro.close()
        return new_task
    monkeypatch.setattr(env, "_schedule_coroutine", fake_schedule)
    env._CLEANUP_TASK = DummyTask(done=True)
    env._REAPER_TASK = DummyTask(done=True, exc=RuntimeError("boom"))
    env.ensure_cleanup_worker()
    assert env._REAPER_TASK is new_task
