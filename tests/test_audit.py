import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from fcntl_compat import LOCK_EX, LOCK_NB, LOCK_UN, flock

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


def test_log_db_access_waits_for_lock(tmp_path: Path) -> None:
    log_file = tmp_path / "blocking.jsonl"
    state_file = Path(f"{log_file}.state")
    state_file.touch()

    with state_file.open("r+") as sf:
        flock(sf.fileno(), LOCK_EX | LOCK_NB)
        thread = threading.Thread(
            target=log_db_access,
            args=("write", "locked", 1, "alpha"),
            kwargs={"log_path": log_file},
        )
        thread.start()
        time.sleep(0.1)
        flock(sf.fileno(), LOCK_UN)
        thread.join(timeout=1)

    assert not thread.is_alive()
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1


def test_log_db_access_bootstrap_skips_on_contention(tmp_path: Path, caplog) -> None:
    log_file = tmp_path / "bootstrap.jsonl"
    state_file = Path(f"{log_file}.state")
    state_file.touch()

    with state_file.open("r+") as sf:
        flock(sf.fileno(), LOCK_EX | LOCK_NB)
        with caplog.at_level(logging.WARNING):
            log_db_access(
                "read",
                "bootstrap_table",
                0,
                "beta",
                log_path=log_file,
                bootstrap_safe=True,
            )
        flock(sf.fileno(), LOCK_UN)

    if log_file.exists():
        assert log_file.read_text() == ""
    assert any("skipping audit log write" in message for message in caplog.messages)
