import json
from audit_logger import log_event
from scripts.purge_visual_agent_audit import purge


def test_purge_visual_agent_run(tmp_path):
    log = tmp_path / "audit_log.jsonl"
    events = [
        {"timestamp": "t1", "event_type": "visual_agent_run", "event_id": "1", "data": {}},
        {"timestamp": "t2", "event_type": "other_event", "event_id": "2", "data": {}},
    ]
    with log.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    purge(str(log))
    remaining = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert len(remaining) == 1
    assert remaining[0]["event_type"] == "other_event"


def test_log_event_blocks_visual_agent(tmp_path):
    log = tmp_path / "audit_log.jsonl"
    event_id = log_event("visual_agent_run", {}, jsonl_path=log)
    assert event_id == ""
    assert not log.exists()
