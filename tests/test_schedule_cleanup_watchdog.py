import asyncio
import ast
import threading
import time
import types
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Iterator

import pytest


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


class DummyLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg, *args, **kwargs):
        return

    def debug(self, msg, *args, **kwargs):
        return

    def error(self, msg, *args, **kwargs):
        self.messages.append(("error", msg % args if args else msg))

    def warning(self, msg, *args, **kwargs):
        self.messages.append(("warning", msg % args if args else msg))

    def exception(self, msg, *args, **kwargs):
        self.messages.append(("exception", msg % args if args else msg))


class GaugeStub:
    def __init__(self):
        self.value = 0.0

    def labels(self, **labels):  # pragma: no cover - simple stub
        return self

    def set(self, value: float) -> None:  # pragma: no cover - simple stub
        self.value = value


class EnvProxy:
    def __init__(self, namespace: dict[str, object]):
        super().__setattr__("_ns", namespace)

    def __getattr__(self, name: str):
        try:
            return self._ns[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        self._ns[name] = value


def _load_environment_subset() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "sandbox_runner" / "environment.py"
    source = path.read_text(encoding="utf8")
    tree = ast.parse(source)
    targets = [
        "_update_worker_heartbeat",
        "_notify_progress",
        "_heartbeat_interval",
        "_background_progress_guard",
        "_progress_heartbeat_scope",
        "_cleanup_worker",
        "_reaper_worker",
        "_clear_watchdog_restart_history",
        "_note_watchdog_restart",
        "_enter_watchdog_cooldown",
        "ensure_cleanup_worker",
        "watchdog_check",
        "schedule_cleanup_check",
        "cancel_cleanup_check",
    ]
    nodes: list[ast.stmt] = []
    for name in targets:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                nodes.append(node)
                break
        else:  # pragma: no cover - safety guard
            raise AssertionError(f"missing function {name}")
    subset = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, object] = {
        "asyncio": asyncio,
        "time": time,
        "threading": threading,
        "Callable": Callable,
        "Iterator": Iterator,
        "contextmanager": contextmanager,
        "_WORKER_CHECK_INTERVAL": 0.1,
        "_POOL_CLEANUP_INTERVAL": 0.1,
        "_CLEANUP_SUBPROCESS_TIMEOUT": 0.5,
        "_HEARTBEAT_GUARD_INTERVAL": 0.05,
        "_HEARTBEAT_GUARD_MAX_DURATION": 1.0,
        "nullcontext": nullcontext,
        "_PROGRESS_SCOPE": threading.local(),
        "_SANDBOX_DISABLE_CLEANUP": False,
        "_log_cleanup_disabled": lambda context=None: None,
        "_suspend_cleanup_workers": lambda reason=None: None,
        "_WATCHDOG_STALL_THRESHOLD": 3,
        "_WATCHDOG_STALL_WINDOW": 10.0,
        "_WATCHDOG_COOLDOWN_SECONDS": 30.0,
        "_WATCHDOG_RECHECK_SECONDS": 15.0,
        "_WATCHDOG_MIN_BASELINE": 5.0,
        "_WATCHDOG_COOLDOWN_UNTIL": None,
        "_WATCHDOG_COOLDOWN_REASON": None,
        "_WATCHDOG_COOLDOWN_LOGGED": False,
        "_WORKER_RESTART_HISTORY": {
            "cleanup": deque(),
            "reaper": deque(),
        },
        "_fallback_logger": lambda: types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        ),
        "_docker_available": lambda: True,
        "deque": deque,
    }
    exec(compile(subset, str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def env():
    ns = _load_environment_subset()
    logger = DummyLogger()
    ns["logger"] = logger
    ns["_fallback_logger"] = lambda: logger
    @contextmanager
    def _noop_scope(callback):
        yield

    ns["_progress_scope"] = _noop_scope
    ns["_get_metrics_module"] = lambda: types.SimpleNamespace(
        cleanup_heartbeat_gauge=GaugeStub(), cleanup_duration_gauge=GaugeStub()
    )
    ns["autopurge_if_needed"] = lambda: None
    ns["ensure_docker_client"] = lambda: None
    ns["reconcile_active_containers"] = lambda: None
    ns["retry_failed_cleanup"] = lambda progress=None: None
    ns["_cleanup_idle_containers"] = lambda *_, **__: (0, 0)
    ns["_purge_stale_vms"] = lambda record_runtime=False: 0
    ns["_prune_volumes"] = lambda progress=None: 0
    ns["_prune_networks"] = lambda progress=None: 0
    ns["report_failed_cleanup"] = lambda alert=False: {}
    def _schedule_stub(coro):
        try:
            coro.close()
        except Exception:
            pass
        return DummyTask()

    ns["_schedule_coroutine"] = _schedule_stub
    ns["_run_cleanup_sync"] = lambda: None
    ns["start_container_event_listener"] = lambda: None
    ns["stop_container_event_listener"] = lambda: None
    ns["_CLEANUP_TASK"] = None
    ns["_REAPER_TASK"] = None
    ns["_DOCKER_CLIENT"] = object()
    ns["_POOL_CLEANUP_INTERVAL"] = 0.1
    ns["_WORKER_CHECK_INTERVAL"] = 0.1
    ns["_CLEANUP_DURATIONS"] = {"cleanup": 0.0, "reaper": 0.0}
    ns["_CLEANUP_CURRENT_RUNTIME"] = {"cleanup": 0.0, "reaper": 0.0}
    ns["_WORKER_ACTIVITY"] = {"cleanup": False, "reaper": False}
    ns["_LAST_CLEANUP_TS"] = time.monotonic()
    ns["_LAST_REAPER_TS"] = time.monotonic()
    ns["_WATCHDOG_METRICS"] = Counter()
    ns["_EVENT_THREAD"] = types.SimpleNamespace(is_alive=lambda: True)
    ns["_WORKER_CHECK_TIMER"] = None
    ns["_CLEANUP_WATCHDOG_MARGIN"] = 5.0
    env = EnvProxy(ns)
    try:
        yield env
    finally:
        timer = env._ns.get("_WORKER_CHECK_TIMER")
        if timer is not None:
            timer.cancel()


def test_watchdog_restarts_cancelled_tasks(env):
    created = []

    def fake_schedule(coro):
        coro.close()
        task = DummyTask()
        created.append(task)
        return task

    env._schedule_coroutine = fake_schedule
    env.start_container_event_listener = lambda: setattr(
        env, "_EVENT_THREAD", types.SimpleNamespace(is_alive=lambda: True)
    )
    env.stop_container_event_listener = lambda: setattr(env, "_EVENT_THREAD", None)
    env._DOCKER_CLIENT = object()
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


def test_watchdog_restarts_stalled_workers(env):
    created = []

    def fake_schedule(coro):
        coro.close()
        task = DummyTask()
        created.append(task)
        return task

    env._schedule_coroutine = fake_schedule
    env.start_container_event_listener = lambda: setattr(
        env, "_EVENT_THREAD", types.SimpleNamespace(is_alive=lambda: True)
    )
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._DOCKER_CLIENT = object()
    env._CLEANUP_TASK = DummyTask()
    env._REAPER_TASK = DummyTask()
    env._LAST_CLEANUP_TS = time.monotonic() - 3 * env._POOL_CLEANUP_INTERVAL
    env._LAST_REAPER_TS = env._LAST_CLEANUP_TS
    env._WATCHDOG_METRICS.clear()
    env._CLEANUP_WATCHDOG_MARGIN = 0.0
    env._WORKER_ACTIVITY["cleanup"] = True
    env._WORKER_ACTIVITY["reaper"] = True

    env.watchdog_check()

    assert env._CLEANUP_TASK in created
    assert env._REAPER_TASK in created
    assert env._WATCHDOG_METRICS["cleanup"] >= 1
    assert env._WATCHDOG_METRICS["reaper"] >= 1
    env._CLEANUP_TASK = None
    env._REAPER_TASK = None


def test_watchdog_ignores_active_cleanup(env):
    interval = 0.05
    env._POOL_CLEANUP_INTERVAL = interval
    autopurge_started = threading.Event()
    autopurge_finished = threading.Event()

    def slow_autopurge():
        autopurge_started.set()
        time.sleep(interval * 1.5)
        autopurge_finished.set()

    env.autopurge_if_needed = slow_autopurge
    env._cleanup_idle_containers = lambda *_, **__: (0, 0)
    env._purge_stale_vms = lambda record_runtime=False: 0
    env._prune_volumes = lambda progress=None: 0
    env._prune_networks = lambda progress=None: 0
    env.report_failed_cleanup = lambda alert=False: {}
    env.ensure_docker_client = lambda: None
    env.reconcile_active_containers = lambda: None
    env.retry_failed_cleanup = lambda progress=None: None
    def schedule_stub(coro):
        coro.close()
        return DummyTask()

    env._schedule_coroutine = schedule_stub
    env._DOCKER_CLIENT = object()
    env._REAPER_TASK = DummyTask()
    env._WATCHDOG_METRICS.clear()
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._LAST_CLEANUP_TS = time.monotonic() - 10 * interval

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    future = asyncio.run_coroutine_threadsafe(env._cleanup_worker(), loop)
    env._CLEANUP_TASK = future

    try:
        assert autopurge_started.wait(timeout=1.0)
        time.sleep(interval)
        env.watchdog_check()
        warnings = [msg for level, msg in env.logger.messages if level == "warning"]
        assert "cleanup worker stalled" not in warnings
        assert autopurge_finished.wait(timeout=1.0)
    finally:
        future.cancel()
        try:
            future.result(timeout=1.0)
        except Exception:
            pass
        env._CLEANUP_TASK = None
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_watchdog_enters_cooldown_after_repeated_stalls(env):
    reasons: list[str | None] = []
    env._WATCHDOG_STALL_THRESHOLD = 2
    env._WATCHDOG_STALL_WINDOW = 120.0
    env._WATCHDOG_COOLDOWN_SECONDS = 45.0
    env._WATCHDOG_RECHECK_SECONDS = 5.0
    env._WORKER_RESTART_HISTORY = {
        "cleanup": deque(maxlen=2),
        "reaper": deque(maxlen=2),
    }
    env._suspend_cleanup_workers = lambda reason=None: reasons.append(reason)
    env._CLEANUP_TASK = DummyTask()
    env._REAPER_TASK = DummyTask()
    env._DOCKER_CLIENT = object()
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._POOL_CLEANUP_INTERVAL = 1.0
    env._CLEANUP_DURATIONS["cleanup"] = 0.0
    env._CLEANUP_WATCHDOG_MARGIN = 0.0

    env._WORKER_ACTIVITY["cleanup"] = True
    env._LAST_CLEANUP_TS = time.monotonic() - 10.0
    env.watchdog_check()

    first_warnings = [msg for level, msg in env.logger.messages if level == "warning"]
    assert "cleanup worker stalled; restarting" in first_warnings

    env.logger.messages.clear()
    env._WORKER_ACTIVITY["cleanup"] = True
    env._LAST_CLEANUP_TS = time.monotonic() - 10.0
    env.watchdog_check()

    assert env._WATCHDOG_COOLDOWN_UNTIL is not None
    assert any(reason and "repeated stalls" in reason for reason in reasons)

    warnings = [msg for level, msg in env.logger.messages if level == "warning"]
    assert "cleanup worker stalled; restarting" not in warnings

    errors = [msg for level, msg in env.logger.messages if level == "error"]
    assert any("repeated stalls" in msg for msg in errors)

def test_cleanup_disabled_short_circuits(env):
    env._ns["_SANDBOX_DISABLE_CLEANUP"] = True
    env.ensure_cleanup_worker()
    assert env._ns["_CLEANUP_TASK"] is None
    assert env._ns["_REAPER_TASK"] is None
    env.watchdog_check()
    assert env.logger.messages == []


def test_watchdog_handles_extended_docker_startup(env):
    interval = 0.05
    env._POOL_CLEANUP_INTERVAL = interval
    env._CLEANUP_WATCHDOG_MARGIN = 0.0
    env._HEARTBEAT_GUARD_INTERVAL = 0.01
    env._HEARTBEAT_GUARD_MAX_DURATION = 5.0

    autopurge_entered = threading.Event()
    release_autopurge = threading.Event()

    def blocking_autopurge():
        autopurge_entered.set()
        release_autopurge.wait(timeout=2.0)

    env.autopurge_if_needed = blocking_autopurge
    env._cleanup_idle_containers = lambda *_, **__: (0, 0)
    env._purge_stale_vms = lambda record_runtime=False: 0
    env._prune_volumes = lambda progress=None: 0
    env._prune_networks = lambda progress=None: 0
    env.report_failed_cleanup = lambda alert=False: {}
    env.ensure_docker_client = lambda: None
    env.reconcile_active_containers = lambda: None
    env.retry_failed_cleanup = lambda progress=None: None
    def schedule_stub(coro):
        coro.close()
        return DummyTask()

    env._schedule_coroutine = schedule_stub
    env._DOCKER_CLIENT = object()
    env._REAPER_TASK = DummyTask()
    env._WATCHDOG_METRICS.clear()
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._LAST_CLEANUP_TS = time.monotonic() - 5 * interval

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    future = asyncio.run_coroutine_threadsafe(env._cleanup_worker(), loop)
    env._CLEANUP_TASK = future

    try:
        assert autopurge_entered.wait(timeout=1.0)
        time.sleep(interval * 6)
        env.watchdog_check()
        warnings = [msg for level, msg in env.logger.messages if level == "warning"]
        assert "cleanup worker stalled" not in warnings
    finally:
        release_autopurge.set()
        future.cancel()
        try:
            future.result(timeout=1.0)
        except Exception:
            pass
        env._CLEANUP_TASK = None
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_watchdog_enforces_minimum_baseline(env):
    env.logger.messages.clear()
    env._POOL_CLEANUP_INTERVAL = 0.0
    env._WORKER_CHECK_INTERVAL = 0.0
    env._WATCHDOG_MIN_BASELINE = 2.5
    env._CLEANUP_WATCHDOG_MARGIN = 0.0
    env._CLEANUP_DURATIONS["cleanup"] = 0.0
    env._CLEANUP_DURATIONS["reaper"] = 0.0
    env._CLEANUP_CURRENT_RUNTIME["cleanup"] = 0.0
    env._CLEANUP_CURRENT_RUNTIME["reaper"] = 0.0
    env._CLEANUP_TASK = DummyTask()
    env._REAPER_TASK = DummyTask()
    env._WORKER_ACTIVITY["cleanup"] = True
    env._WORKER_ACTIVITY["reaper"] = True
    env._LAST_CLEANUP_TS = time.monotonic() - 0.5
    env._LAST_REAPER_TS = time.monotonic() - 0.5

    env.watchdog_check()

    warnings = [msg for level, msg in env.logger.messages if level == "warning"]
    assert "cleanup worker stalled" not in warnings
    assert "reaper worker stalled" not in warnings


def test_watchdog_accepts_midpass_heartbeats(env):
    interval = 0.2
    delay = 0.75 * interval
    env._POOL_CLEANUP_INTERVAL = interval
    env._DOCKER_CLIENT = object()
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._REAPER_TASK = DummyTask()
    env._CLEANUP_TASK = None
    env._WATCHDOG_METRICS.clear()
    env._LAST_CLEANUP_TS = time.monotonic()

    events = {
        name: threading.Event()
        for name in (
            "retry",
            "cleanup",
            "purge_vms",
            "prune_volumes",
            "prune_networks",
            "report",
        )
    }

    def slow_retry(progress=None) -> None:
        events["retry"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()

    def slow_cleanup_idle_containers(*, progress=None) -> tuple[int, int]:
        events["cleanup"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()
        return (0, 0)

    def slow_purge_vms(record_runtime: bool = False) -> int:  # noqa: FBT001
        events["purge_vms"].set()
        time.sleep(delay)
        return 0

    def slow_prune_volumes(progress=None) -> int:
        events["prune_volumes"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()
        return 0

    def slow_prune_networks(progress=None) -> int:
        events["prune_networks"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()
        return 0

    def slow_report_failed_cleanup(alert: bool = False) -> dict[str, object]:  # noqa: FBT001
        events["report"].set()
        time.sleep(delay)
        return {}

    env.retry_failed_cleanup = slow_retry
    env._cleanup_idle_containers = slow_cleanup_idle_containers
    env._purge_stale_vms = slow_purge_vms
    env._prune_volumes = slow_prune_volumes
    env._prune_networks = slow_prune_networks
    env.report_failed_cleanup = slow_report_failed_cleanup

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def schedule(coro):
        name = getattr(getattr(coro, "cr_code", None), "co_name", "")
        if name == "_cleanup_worker":
            return asyncio.run_coroutine_threadsafe(coro, loop)
        coro.close()
        return DummyTask()

    env._schedule_coroutine = schedule

    stop = threading.Event()

    env.ensure_cleanup_worker()

    def watchdog_runner() -> None:
        while not stop.is_set():
            env.watchdog_check()
            time.sleep(interval / 4)

    watchdog_thread = threading.Thread(target=watchdog_runner, daemon=True)
    watchdog_thread.start()

    try:
        for event in events.values():
            assert event.wait(timeout=5.0)

        warnings = [msg for level, msg in env.logger.messages if level == "warning"]
        assert "cleanup worker stalled; restarting" not in warnings
    finally:
        stop.set()
        watchdog_thread.join(timeout=1.0)
        task = env._CLEANUP_TASK
        if task is not None:
            task.cancel()
            try:
                task.result(timeout=1.0)
            except Exception:
                pass
        env._CLEANUP_TASK = None
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_watchdog_accepts_reaper_midpass_heartbeats(env):
    interval = 0.2
    delay = 0.75 * interval
    env._POOL_CLEANUP_INTERVAL = interval
    env._DOCKER_CLIENT = object()
    env._EVENT_THREAD = types.SimpleNamespace(is_alive=lambda: True)
    env._CLEANUP_TASK = DummyTask()
    env._WATCHDOG_METRICS.clear()
    env._LAST_CLEANUP_TS = time.monotonic()
    env._LAST_REAPER_TS = time.monotonic()

    events = {
        name: threading.Event()
        for name in (
            "autopurge",
            "reconcile",
            "reap",
            "purge_vms",
            "prune_volumes",
            "prune_networks",
        )
    }

    def slow_autopurge():
        events["autopurge"].set()
        time.sleep(delay)

    def slow_reconcile():
        events["reconcile"].set()
        time.sleep(delay)

    def slow_reap(progress=None) -> int:
        events["reap"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()
        return 0

    def slow_purge_vms(record_runtime: bool = True) -> int:  # noqa: FBT001
        events["purge_vms"].set()
        time.sleep(delay)
        return 0

    def slow_prune_volumes(progress=None) -> int:
        events["prune_volumes"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()
        return 0

    def slow_prune_networks(progress=None) -> int:
        events["prune_networks"].set()
        if progress is not None:
            progress()
        time.sleep(delay)
        if progress is not None:
            progress()
        return 0

    env.autopurge_if_needed = slow_autopurge
    env.reconcile_active_containers = slow_reconcile
    env._reap_orphan_containers = slow_reap
    env._purge_stale_vms = slow_purge_vms
    env._prune_volumes = slow_prune_volumes
    env._prune_networks = slow_prune_networks

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def schedule(coro):
        name = getattr(getattr(coro, "cr_code", None), "co_name", "")
        if name == "_reaper_worker":
            return asyncio.run_coroutine_threadsafe(coro, loop)
        coro.close()
        return DummyTask()

    env._schedule_coroutine = schedule

    env._REAPER_TASK = schedule(env._reaper_worker())

    stop = threading.Event()

    env.ensure_cleanup_worker()

    def watchdog_runner() -> None:
        while not stop.is_set():
            env.watchdog_check()
            time.sleep(interval / 4)

    watchdog_thread = threading.Thread(target=watchdog_runner, daemon=True)
    watchdog_thread.start()

    try:
        for event in events.values():
            assert event.wait(timeout=5.0)

        warnings = [msg for level, msg in env.logger.messages if level == "warning"]
        assert "reaper worker stalled; restarting" not in warnings
    finally:
        stop.set()
        watchdog_thread.join(timeout=1.0)
        task = env._REAPER_TASK
        if task is not None:
            task.cancel()
            try:
                task.result(timeout=1.0)
            except Exception:
                pass
        env._REAPER_TASK = None
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()
