import json
import sqlite3

from audit import log_db_access


def test_log_db_access_inserts_into_db(tmp_path):
    log_file = tmp_path / "shared_db_access.log"
    conn = sqlite3.connect(tmp_path / "audit.db")

    log_db_access(
        "write",
        "telemetry",
        3,
        "beta",
        log_path=log_file,
        db_conn=conn,
    )

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert entries[0]["action"] == "write"

    cur = conn.cursor()
    cur.execute('SELECT action, "table", rows, menace_id FROM shared_db_audit')
    assert cur.fetchone() == ("write", "telemetry", 3, "beta")
    conn.close()
