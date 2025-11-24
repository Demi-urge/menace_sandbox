import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import audit

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


def test_log_db_access_can_disable_file_logging(tmp_path: Path, monkeypatch) -> None:
    log_file = tmp_path / "disabled.jsonl"
    monkeypatch.setenv("DB_AUDIT_DISABLE_FILE", "1")

    log_db_access("read", "disabled", 0, "alpha", log_path=log_file)

    assert not log_file.exists()
    assert not Path(f"{log_file}.state").exists()


def test_log_db_access_bootstrap_avoids_state_file_on_windows(
    tmp_path: Path, monkeypatch
) -> None:
    log_file = tmp_path / "unwritable" / "slow.jsonl"

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def slow_open(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        time.sleep(0.2)
        raise AssertionError("os.open should not be called during bootstrap-safe logging")

    monkeypatch.setattr(audit.os, "open", slow_open)
    monkeypatch.setattr(audit.os, "name", "nt")

    start = time.perf_counter()
    log_db_access(
        "read",
        "windows_bootstrap",
        0,
        "omega",
        log_path=log_file,
        bootstrap_safe=True,
    )
    elapsed = time.perf_counter() - start

    assert calls == []
    assert elapsed < 0.1
    assert not log_file.exists()
    assert not Path(f"{log_file}.state").exists()
