import json

import db_router

from audit import log_db_access


def test_log_db_access_inserts_into_db(tmp_path):
    log_file = tmp_path / "shared_db_access.log"
    local_db = tmp_path / "local.db"
    shared_db = tmp_path / "shared.db"

    router = db_router.init_db_router("beta", str(local_db), str(shared_db))
    try:
        log_db_access(
            "write",
            "telemetry",
            3,
            "beta",
            log_to_db=True,
            log_path=str(log_file),
        )
        entries = [json.loads(line) for line in log_file.read_text().splitlines()]
        assert entries[0]["action"] == "write"
        cur = router.local_conn.cursor()
        cur.execute(
            "SELECT action, table_name, row_count, menace_id FROM db_access_audit"
        )
        assert cur.fetchone() == ("write", "telemetry", 3, "beta")
    finally:
        router.close()
        db_router.GLOBAL_ROUTER = None
