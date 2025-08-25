import json

from db_write_buffer import append_to_queue, buffer_shared_insert


def test_append_to_queue_writes_record(tmp_path):
    append_to_queue("foo", {"x": 1}, "m1", hash_fields=["x"], queue_dir=tmp_path)
    queue_file = tmp_path / "foo_queue.jsonl"
    data = queue_file.read_text(encoding="utf-8")
    assert data.endswith("\n")
    rec = json.loads(data)
    assert rec == {
        "table": "foo",
        "values": {"x": 1},
        "source_menace_id": "m1",
        "hash_fields": ["x"],
    }


def test_buffer_shared_insert_enriches_menace_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_ID", "alpha")
    buffer_shared_insert("bar", {"y": 2}, ["y"], queue_dir=tmp_path)
    queue_file = tmp_path / "bar_queue.jsonl"
    rec = json.loads(queue_file.read_text())
    assert rec["source_menace_id"] == "alpha"
    assert rec["hash_fields"] == ["y"]
    assert rec["values"] == {"y": 2}
