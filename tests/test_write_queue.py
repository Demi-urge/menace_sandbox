import json
import threading

from db_dedup import compute_content_hash
from db_write_queue import append_record, read_queue, remove_processed_lines


def test_append_read_remove(tmp_path):
    """Records can be appended, read back and removed from queue files."""
    queue_dir = tmp_path
    append_record("example", {"a": 1}, "m1", queue_dir)
    append_record("example", {"b": 2}, "m1", queue_dir)

    path = queue_dir / "m1.jsonl"
    records = read_queue(path)
    expected = [
        {
            "table": "example",
            "data": {"a": 1},
            "source_menace_id": "m1",
            "hash": compute_content_hash({"a": 1}),
            "hash_fields": ["a"],
        },
        {
            "table": "example",
            "data": {"b": 2},
            "source_menace_id": "m1",
            "hash": compute_content_hash({"b": 2}),
            "hash_fields": ["b"],
        },
    ]
    assert records == expected

    remove_processed_lines(path, 1)
    assert read_queue(path) == [records[1]]

    backup = queue_dir / "queue.log.bak"
    with backup.open("r", encoding="utf-8") as fh:
        bak_records = [json.loads(line) for line in fh if line.strip()]
    assert bak_records == [expected[0]]


def test_threaded_writes_are_atomic(tmp_path):
    """File locks prevent interleaved writes from multiple threads."""
    queue_dir = tmp_path
    path = queue_dir / "m1.jsonl"

    def worker(n: int) -> None:
        append_record("example", {"n": n}, "m1", queue_dir)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    payloads = [json.loads(line)["data"]["n"] for line in lines]
    assert set(payloads) == set(range(20))
