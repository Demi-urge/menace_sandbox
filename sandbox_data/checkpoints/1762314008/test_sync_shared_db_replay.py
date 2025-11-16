import json
import sync_shared_db


def test_replay_failed_requeues_and_preserves(tmp_path):
    queue_dir = tmp_path
    failed = queue_dir / "queue.failed.jsonl"

    lines = [
        '{"record": {"menace_id": "m1", "foo": 1}}\n',
        '{"record": "{\\"menace_id\\": \\\"m2\\\", \\\"foo\\\": 2}"}\n',
        '{"record": {"bar": 3}}\n',
        '{"record": 123}\n',
        '{"record": "{invalid}"}\n',
        'not-json\n',
    ]
    failed.write_text("".join(lines), encoding="utf-8")

    sync_shared_db.replay_failed(queue_dir)

    # failed file renamed to .bak
    backup = queue_dir / "queue.failed.jsonl.bak"
    assert backup.exists()
    assert not failed.exists()
    assert backup.read_text(encoding="utf-8") == "".join(lines)

    # valid records requeued to menace id or replay file
    m1_path = queue_dir / "m1.jsonl"
    m2_path = queue_dir / "m2.jsonl"
    replay_path = queue_dir / "replay.jsonl"

    assert m1_path.read_text(encoding="utf-8") == json.dumps(
        {"foo": 1, "menace_id": "m1"}, sort_keys=True
    ) + "\n"
    assert m2_path.read_text(encoding="utf-8") == json.dumps(
        {"foo": 2, "menace_id": "m2"}, sort_keys=True
    ) + "\n"
    assert replay_path.read_text(encoding="utf-8") == json.dumps(
        {"bar": 3}, sort_keys=True
    ) + "\n"
