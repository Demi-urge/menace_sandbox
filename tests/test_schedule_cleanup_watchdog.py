import asyncio
import ast
import threading
import time
import types
from collections import Counter
from pathlib import Path

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
        "_cleanup_worker",
        "_reaper_worker",
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
        "_WORKER_CHECK_INTERVAL": 0.1,
        "_POOL_CLEANUP_INTERVAL": 0.1,
    }
    exec(compile(subset, str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def env():
    ns = _load_environment_subset()
    logger = DummyLogger()
    ns["logger"] = logger
    ns["_get_metrics_module"] = lambda: types.SimpleNamespace(
        cleanup_heartbeat_gauge=GaugeStub(), cleanup_duration_gauge=GaugeStub()
    )
    ns["autopurge_if_needed"] = lambda: None
    ns["ensure_docker_client"] = lambda: None
    ns["reconcile_active_containers"] = lambda: None
    ns["retry_failed_cleanup"] = lambda: None
    ns["_cleanup_idle_containers"] = lambda: (0, 0)
    ns["_purge_stale_vms"] = lambda record_runtime=False: 0
    ns["_prune_volumes"] = lambda: 0
    ns["_prune_networks"] = lambda: 0
    ns["report_failed_cleanup"] = lambda alert=False: {}
    ns["_schedule_coroutine"] = lambda coro: DummyTask()
    ns["_run_cleanup_sync"] = lambda: None
    ns["start_container_event_listener"] = lambda: None
    ns["stop_container_event_listener"] = lambda: None
    ns["_CLEANUP_TASK"] = None
    ns["_REAPER_TASK"] = None
    ns["_DOCKER_CLIENT"] = object()
    ns["_POOL_CLEANUP_INTERVAL"] = 0.1
    ns["_WORKER_CHECK_INTERVAL"] = 0.1
    ns["_CLEANUP_DURATIONS"] = {"cleanup": 0.0, "reaper": 0.0}
    ns["_LAST_CLEANUP_TS"] = time.monotonic()
    ns["_LAST_REAPER_TS"] = time.monotonic()
    ns["_WATCHDOG_METRICS"] = Counter()
    ns["_EVENT_THREAD"] = types.SimpleNamespace(is_alive=lambda: True)
    ns["_WORKER_CHECK_TIMER"] = None
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
    env._cleanup_idle_containers = lambda: (0, 0)
    env._purge_stale_vms = lambda record_runtime=False: 0
    env._prune_volumes = lambda: 0
    env._prune_networks = lambda: 0
    env.report_failed_cleanup = lambda alert=False: {}
    env.ensure_docker_client = lambda: None
    env.reconcile_active_containers = lambda: None
    env.retry_failed_cleanup = lambda: None
    env._schedule_coroutine = lambda coro: DummyTask()
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
