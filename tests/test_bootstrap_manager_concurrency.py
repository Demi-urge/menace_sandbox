import logging
import threading
import time

import pytest

from bootstrap_manager import BootstrapManager


@pytest.fixture
def manager():
    return BootstrapManager()


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
