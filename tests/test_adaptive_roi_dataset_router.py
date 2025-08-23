from menace.adaptive_roi_dataset import _collect_error_history
from menace.db_router import init_db_router
from unittest.mock import patch

def test_error_history_uses_router(tmp_path):
    router = init_db_router("eh", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    conn = router.get_connection("telemetry")
    conn.execute("CREATE TABLE telemetry(ts TEXT, resolution_status TEXT)")
    conn.commit()
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        _collect_error_history(tmp_path / "errors.db", router=router)
        gc.assert_called_with("telemetry")
