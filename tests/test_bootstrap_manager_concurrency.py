import logging
import threading
import time

import pytest

from bootstrap_manager import BootstrapManager


@pytest.fixture
def manager():
    return BootstrapManager()


@pytest.fixture
def slow_bootstrapper():
    """Return a factory that produces slow/optionally failing bootstrap callables."""

    def _factory(*, fail_first: bool = False, delay: float = 0.05):
        class _SlowBootstrap:
            def __init__(self) -> None:
                self.calls = 0
                self.fail_first = fail_first
                self.delay = delay
                self.started = threading.Event()

            def __call__(self) -> str:
                self.calls += 1
                self.started.set()
                time.sleep(self.delay)
                if self.fail_first and self.calls == 1:
                    raise RuntimeError("bootstrap failed")
                return f"bootstrapped-{self.calls}"

        return _SlowBootstrap()

    return _factory


def test_run_once_bootstraps_only_once_across_modules(manager, caplog):
    caplog.set_level(logging.INFO)
    execution_count = 0
    results: list[tuple[str, object]] = []
    barrier = threading.Barrier(3)

    def bootstrap_task():
        nonlocal execution_count
        time.sleep(0.05)
        execution_count += 1
        return f"bootstrap-{execution_count}"

    def module_caller(module_name: str) -> None:
        barrier.wait()
        result = manager.run_once(
            "shared_bootstrap",
            bootstrap_task,
            logger=logging.getLogger(module_name),
        )
        results.append((module_name, result))

    threads = [
        threading.Thread(target=module_caller, args=("ResearchAggregator",)),
        threading.Thread(target=module_caller, args=("PredictionManager",)),
        threading.Thread(target=module_caller, args=("VectorService",)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert execution_count == 1
    assert {result for _, result in results} == {"bootstrap-1"}

    starts = [record for record in caplog.records if record.getMessage() == "bootstrap helper starting"]
    finishes = [record for record in caplog.records if record.getMessage() == "bootstrap helper finished"]
    already_running = [
        record
        for record in caplog.records
        if record.getMessage() == "bootstrap helper already running"
    ]

    assert len(starts) == 1
    assert len(finishes) == 1
    assert len(already_running) == 2

    cached_result = manager.run_once(
        "shared_bootstrap",
        bootstrap_task,
        logger=logging.getLogger("VectorService"),
    )
    assert cached_result == "bootstrap-1"
    cached_logs = [
        record for record in caplog.records if record.getMessage() == "bootstrap helper skipped (cached)"
    ]
    assert len(cached_logs) == 1


def test_wait_until_ready_returns_cached_state(manager):
    call_count = 0

    def readiness_probe() -> bool:
        nonlocal call_count
        call_count += 1
        return True

    assert manager.wait_until_ready(timeout=1.0, check=readiness_probe)
    assert call_count == 1

    for _ in range(3):
        assert manager.wait_until_ready(timeout=1.0, check=readiness_probe)

    assert call_count == 1


def test_reentrancy_guard_skips_redundant_calls(manager, slow_bootstrapper):
    bootstrap = slow_bootstrapper()

    first = manager.run_once("reentrant-step", bootstrap)
    second = manager.run_once("reentrant-step", bootstrap)

    assert first == second == "bootstrapped-1"
    assert bootstrap.calls == 1


def test_already_bootstrapped_state_respected_by_waiters(manager):
    manager.mark_ready()
    called = False

    def readiness_probe() -> bool:
        nonlocal called
        called = True
        raise AssertionError("readiness probe should not run after ready")

    results = []

    def waiter() -> None:
        results.append(manager.wait_until_ready(timeout=0.5, check=readiness_probe))

    threads = [threading.Thread(target=waiter) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True, True, True]
    assert not called


def test_health_check_path_avoids_recursion(manager):
    calls: list[float] = []

    def readiness_probe() -> bool:
        calls.append(time.monotonic())
        manager.mark_ready()
        return True

    assert manager.wait_until_ready(timeout=1.0, check=readiness_probe)
    assert len(calls) == 1


def test_concurrency_survives_failed_bootstrap_then_recovers(manager, slow_bootstrapper):
    bootstrap = slow_bootstrapper(fail_first=True, delay=0.02)

    with pytest.raises(RuntimeError):
        manager.run_once("critical-step", bootstrap)

    bootstrap.fail_first = False
    barrier = threading.Barrier(2)
    results: list[str] = []

    def caller():
        barrier.wait()
        results.append(
            manager.run_once("critical-step", bootstrap, logger=logging.getLogger("caller"))
        )

    threads = [threading.Thread(target=caller) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["bootstrapped-2", "bootstrapped-2"]
    assert bootstrap.calls == 2
