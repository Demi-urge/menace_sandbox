import json
from menace import violation_logger as vl


def test_log_and_load(tmp_path, monkeypatch):
    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    vl.log_violation("abc", "policy", 3, {"detail": "bad"})
    vl.log_violation("def", "security", 5, {"detail": "worse"})

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["entry_id"] == "abc"

    recents = vl.load_recent_violations(1)
    assert recents and recents[0]["entry_id"] == "def"

    summary = vl.violation_summary("abc")
    assert "policy" in summary
