import json

import pytest

from tests.test_visual_agent_server import _setup_va


def test_checksum_recovers_tampered_file(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()

    path = tmp_path / "visual_agent_queue.json"
    data = json.loads(path.read_text())
    data["queue"][0]["prompt"] = "bad"
    path.write_text(json.dumps(data))
    # do not modify checksum file -> mismatch

    va2 = _setup_va(monkeypatch, tmp_path)
    assert not va2.task_queue
    assert not va2.job_status
    bak1 = tmp_path / "visual_agent_queue.json.bak1"
    assert bak1.exists()
    assert "bad" in bak1.read_text()


def test_backup_rotation(monkeypatch, tmp_path):
    content = []
    for i in range(4):
        bad = f"bad{i}"
        (tmp_path / "visual_agent_queue.json").write_text(bad)
        if (tmp_path / "visual_agent_queue.json.sha256").exists():
            (tmp_path / "visual_agent_queue.json.sha256").unlink()
        va = _setup_va(monkeypatch, tmp_path)
        content.append(bad)

    backups = sorted(tmp_path.glob("visual_agent_queue.json.bak*"))
    assert len(backups) == 3
    assert backups[0].read_text() == "bad3"
    assert backups[1].read_text() == "bad2"
    assert backups[2].read_text() == "bad1"
    # final queue file should be reset
    assert json.loads((tmp_path / "visual_agent_queue.json").read_text()) == {"queue": [], "status": {}}
