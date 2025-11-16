import json
import threading
import hashlib
from pathlib import Path

from audit import log_db_access


def test_log_db_access(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"

    def writer(idx: int) -> None:
        log_db_access("read", f"table{idx}", idx, f"id{idx}", log_path=log_file)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["table"] for r in records} == {"table0", "table1"}
    assert all(r["action"] == "read" for r in records)
    prev_hash = "0" * 64
    for rec in records:
        assert "hash" in rec
        data = {k: v for k, v in rec.items() if k != "hash"}
        expected = hashlib.sha256(
            (prev_hash + json.dumps(data, sort_keys=True)).encode()
        ).hexdigest()
        assert rec["hash"] == expected
        prev_hash = rec["hash"]

    state = Path(f"{log_file}.state").read_text().strip()
    assert state == prev_hash
