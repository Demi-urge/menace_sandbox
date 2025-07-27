import time
import types
import threading

import pytest

from tests.test_visual_agent_server import _setup_va

pytest.importorskip("fastapi")

def test_watchdog_recovers_dead_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("VA_WATCHDOG_INTERVAL", "0.01")
    va = _setup_va(monkeypatch, tmp_path, start_worker=True)
    processed = []
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: processed.append("ok"))
    original_qw = va._queue_worker
    def crash_worker():
        return
    monkeypatch.setattr(va, "_queue_worker", crash_worker)
    va._start_background_threads()
    time.sleep(0.05)
    monkeypatch.setattr(va, "_queue_worker", original_qw)
    with va._queue_lock:
        va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
        va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    for _ in range(100):
        if processed:
            break
        time.sleep(0.05)
    va._exit_event.set()
    if va._worker_thread and va._worker_thread.is_alive():
        va._worker_thread.join(timeout=1)
    if va._watchdog_thread and va._watchdog_thread.is_alive():
        va._watchdog_thread.join(timeout=1)
    assert processed == ["ok"]

def test_watchdog_recovers_corrupt_db(monkeypatch, tmp_path):
    monkeypatch.setenv("VA_WATCHDOG_INTERVAL", "0.01")
    va = _setup_va(monkeypatch, tmp_path, start_worker=True)
    processed = []
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: processed.append("ok"))
    original_qw = va._queue_worker
    def crash_worker():
        return
    monkeypatch.setattr(va, "_queue_worker", crash_worker)
    va._start_background_threads()
    time.sleep(0.05)
    monkeypatch.setattr(va, "_queue_worker", original_qw)
    va.QUEUE_DB.write_text("bad")
    with va._queue_lock:
        va.job_status["b"] = {"status": "queued", "prompt": "p", "branch": None}
        va.task_queue.append({"id": "b", "prompt": "p", "branch": None})
    for _ in range(100):
        if processed:
            break
        time.sleep(0.05)
    backups = list(tmp_path.glob("visual_agent_queue.db.corrupt.*"))
    va._exit_event.set()
    if va._worker_thread and va._worker_thread.is_alive():
        va._worker_thread.join(timeout=1)
    if va._watchdog_thread and va._watchdog_thread.is_alive():
        va._watchdog_thread.join(timeout=1)
    assert processed == ["ok"]
    assert backups
