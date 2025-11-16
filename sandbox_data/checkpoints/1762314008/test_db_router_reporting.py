from __future__ import annotations

import sys
import time
import types

from db_router import DBRouter


def test_periodic_reporting_flushes(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared.db"
    local_db = tmp_path / "local.db"
    calls: list[tuple[str, str, str, int]] = []
    dummy = types.SimpleNamespace(
        record_table_access=lambda menace, table, op, count=1: calls.append(
            (menace, table, op, count)
        )
    )
    monkeypatch.setitem(sys.modules, "telemetry_backend", dummy)

    router = DBRouter("test", str(local_db), str(shared_db))
    try:
        router.get_connection("bots")
        router.start_periodic_reporting(interval=0.1)
        time.sleep(0.15)
    finally:
        router.close()

    assert ("test", "bots", "shared", 1) in calls
