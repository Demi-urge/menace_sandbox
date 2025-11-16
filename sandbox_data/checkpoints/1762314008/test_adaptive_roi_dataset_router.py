from menace.adaptive_roi_dataset import _collect_error_history
from menace.db_router import init_db_router
from unittest.mock import patch


def test_error_history_uses_router(tmp_path):
    router = init_db_router("eh", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    conn = router.get_connection("telemetry")
    conn.execute(
        "CREATE TABLE telemetry(ts TEXT, resolution_status TEXT, source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO telemetry(ts, resolution_status, source_menace_id) VALUES (?,?,?)",
        [
            ("2020-01-01T00:00:00", "resolved", "m1"),
            ("2020-01-02T00:00:00", "resolved", "m2"),
        ],
    )
    conn.commit()
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        local = _collect_error_history(
            tmp_path / "errors.db", router=router, scope="local", source_menace_id="m1"
        )
        global_rec = _collect_error_history(
            tmp_path / "errors.db", router=router, scope="global", source_menace_id="m1"
        )
        all_rec = _collect_error_history(
            tmp_path / "errors.db", router=router, scope="all", source_menace_id="m1"
        )
        gc.assert_called_with("telemetry")
    assert len(local) == 1
    assert len(global_rec) == 1
    assert len(all_rec) == 2
