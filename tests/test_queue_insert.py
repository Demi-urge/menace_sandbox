import json

from db_write_queue import queue_insert


def test_queue_insert_writes_record(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_ID", "m1")
    queue_insert("foo", {"x": 1}, ["x"], queue_path=tmp_path)
    queue_file = tmp_path / "foo_queue.jsonl"
    lines = queue_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec == {
        "table": "foo",
        "op": "insert",
        "data": {"x": 1},
        "hash": rec["hash"],
        "hash_fields": ["x"],
        "source_menace_id": "m1",
    }


def test_queue_insert_enriches_menace_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MENACE_ID", "alpha")
    queue_insert("bar", {"y": 2}, ["y"], queue_path=tmp_path)
    queue_file = tmp_path / "bar_queue.jsonl"
    rec = json.loads(queue_file.read_text().splitlines()[0])
    assert rec["source_menace_id"] == "alpha"
    assert rec["hash_fields"] == ["y"]
    assert rec["data"] == {"y": 2}
