import json

from tests.test_visual_agent_server import _setup_va
from visual_agent_queue import VisualAgentQueue


def test_corrupt_db_resets_queue(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    va.task_queue.append({"id": "a", "prompt": "p"})

    db_path = tmp_path / "visual_agent_queue.db"
    db_path.write_text("bad")

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()
    assert not list(va2.task_queue)
    assert not va2.job_status


def test_migrate_creates_backup(monkeypatch, tmp_path):
    qfile = tmp_path / "visual_agent_queue.jsonl"
    qfile.write_text(json.dumps({"id": "x"}) + "\n")
    db = tmp_path / "visual_agent_queue.db"

    VisualAgentQueue.migrate_from_jsonl(db, qfile)
    assert qfile.with_suffix(qfile.suffix + ".bak").exists()

    VisualAgentQueue.migrate_from_jsonl(db, qfile)
    backups = list(tmp_path.glob("visual_agent_queue.jsonl.bak*"))
    assert len(backups) == 1
