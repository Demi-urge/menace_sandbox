import importlib
import importlib.util
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("dynamic_path_router", ROOT / "dynamic_path_router.py")
dynamic_path_router = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(dynamic_path_router)  # type: ignore[union-attr]
sys.modules["dynamic_path_router"] = dynamic_path_router

env = importlib.import_module("menace_sandbox.sandbox_runner.environment")
sys.modules["sandbox_runner.environment"] = env


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

def test_watchdog_recovers_event_listener_failure(monkeypatch):
    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())

    calls = []
    def fake_start_listener():
        calls.append(True)
        env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)

    monkeypatch.setattr(env, "start_container_event_listener", fake_start_listener)

    env._EVENT_THREAD = None
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

    assert calls
    assert env._WATCHDOG_METRICS.get("event", 0) >= 1
    assert env._WORKER_CHECK_TIMER is not timer

    env.cancel_cleanup_check()


def test_cleanup_timeout_does_not_stall_watchdog(monkeypatch, tmp_path, caplog):
    caplog.set_level("WARNING")
    file = tmp_path / "cleanup.json"
    file.write_text("{}")
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)

    monkeypatch.setattr(env, "autopurge_if_needed", lambda: None)
    monkeypatch.setattr(env, "ensure_docker_client", lambda: None)
    monkeypatch.setattr(env, "ensure_cleanup_worker", lambda: None)
    monkeypatch.setattr(env, "retry_failed_cleanup", lambda progress=None: (0, 0))
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda: (0, 0))
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)
    monkeypatch.setattr(env, "_PRUNE_VOLUMES", False)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", False)

    call_count = {"value": 0}

    def fake_run(cmd, *args, **kwargs):
        if call_count["value"] == 0:
            call_count["value"] += 1
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))
        call_count["value"] += 1
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env._WATCHDOG_METRICS.clear()
    env._CLEANUP_TASK = None
    env._REAPER_TASK = None
    env._DOCKER_CLIENT = object()
    env._LAST_CLEANUP_TS = time.monotonic() - 100
    env._LAST_REAPER_TS = time.monotonic()

    env._run_cleanup_sync()

    assert "timed out" in caplog.text
    assert time.monotonic() - env._LAST_CLEANUP_TS < 10

    before = dict(env._WATCHDOG_METRICS)
    env.watchdog_check()
    after = dict(env._WATCHDOG_METRICS)

    assert after.get("cleanup", 0) == before.get("cleanup", 0)
