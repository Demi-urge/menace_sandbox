import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_visual_agent_server import _setup_va
from pathlib import Path


def test_worker_recovers_corrupt_db(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path, start_worker=True)
    va._initialize_state()
    va._start_background_threads()

    finished = threading.Event()
    monkeypatch.setattr(va, "run_menace_pipeline", lambda *a, **k: finished.set())

    with va._queue_lock:
        va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
        va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()

    (tmp_path / "visual_agent_queue.db").write_text("bad")

    finished.wait(1)
    va._exit_event.set()
    if va._worker_thread:
        va._worker_thread.join(timeout=1)

    backups = list(tmp_path.glob("visual_agent_queue.db.corrupt.*"))
    assert finished.is_set()
    assert backups
    assert va.job_status["a"]["status"] == "completed"
