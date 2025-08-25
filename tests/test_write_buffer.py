import json
from db_write_buffer import buffer_shared_insert


def test_buffer_shared_insert_writes_valid_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_ID", "tester")
    buffer_shared_insert("foo", {"x": 1}, ["x"], queue_dir=tmp_path)
    buffer_shared_insert("foo", {"x": 2}, ["x"], queue_dir=tmp_path)
    queue_file = tmp_path / "foo_queue.jsonl"
    content = queue_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["values"] == {"x": 1}
    assert records[1]["values"] == {"x": 2}
    assert all(r["table"] == "foo" for r in records)
    assert all(r["source_menace_id"] == "tester" for r in records)
