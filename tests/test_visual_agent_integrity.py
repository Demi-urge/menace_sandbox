import json
import time
from pathlib import Path
import pytest
from tests.test_visual_agent_auto_recover import _setup_va
from visual_agent_queue import VisualAgentQueue


def test_corrupt_db_auto_recovery(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    db_path = tmp_path / "visual_agent_queue.db"
    db_path.write_text("garbage")
    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()
    backups = list(tmp_path.glob("visual_agent_queue.db.corrupt.*"))
    assert backups
    assert not list(va2.task_queue)
    assert not va2.job_status


def test_restart_persists_and_requeues(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    va.job_status.clear()
    va.task_queue.clear()
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va.task_queue.append({"id": "b", "prompt": "q", "branch": None})
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.job_status["b"] = {"status": "running", "prompt": "q", "branch": None}
    va._persist_state()

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()
    ids = [t["id"] for t in va2.task_queue.load_all()]
    assert set(ids) == {"a", "b"}
    assert va2.job_status["b"]["status"] == "queued"


def test_integrity_endpoint_recovers(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    import asyncio

    db_path = tmp_path / "visual_agent_queue.db"
    db_path.write_text("bad")
    resp = asyncio.run(va.queue_integrity(x_token=va.API_TOKEN))
    assert resp["rebuilt"]
    backups = list(tmp_path.glob("visual_agent_queue.db.corrupt.*"))
    assert backups
