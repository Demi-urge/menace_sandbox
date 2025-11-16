import json
import sqlite3
import hashlib

from pathlib import Path

from audit import log_db_access


def test_log_db_access_inserts_into_db(tmp_path):
    log_file = tmp_path / "shared_db_access.log"
    conn = getattr(sqlite3, "connect")(tmp_path / "audit.db")  # noqa: SQL001

    log_db_access(
        "write",
        "telemetry",
        3,
        "beta",
        log_path=log_file,
        db_conn=conn,
        log_to_db=True,
    )

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert entries[0]["action"] == "write"
    prev_hash = "0" * 64
    rec = entries[0]
    assert "hash" in rec
    data = {k: v for k, v in rec.items() if k != "hash"}
    expected = hashlib.sha256((prev_hash + json.dumps(data, sort_keys=True)).encode()).hexdigest()
    assert rec["hash"] == expected
    state = Path(f"{log_file}.state").read_text().strip()
    assert state == rec["hash"]

    cur = conn.cursor()
    cur.execute('SELECT action, "table", rows, menace_id FROM shared_db_audit')
    assert cur.fetchone() == ("write", "telemetry", 3, "beta")
    conn.close()
