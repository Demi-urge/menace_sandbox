from unittest.mock import patch

import db_router
from patch_safety import PatchSafety


def test_patch_safety_uses_router(tmp_path):
    local_db = tmp_path / "failures.db"
    router = db_router.DBRouter("ps", str(local_db), str(tmp_path / "shared.db"))
    conn = router.get_connection("failures", operation="write")
    conn.execute(
        """CREATE TABLE failures (
            cause TEXT,
            demographics TEXT,
            profitability REAL,
            retention REAL,
            cac REAL,
            roi REAL
        )"""
    )
    conn.commit()
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        ps = PatchSafety(storage_path=None, failure_db_path=str(local_db), router=router)
        ps.load_failures(force=True)
        gc.assert_called_with("failures")
    assert "failures" in db_router.LOCAL_TABLES
    router.close()
