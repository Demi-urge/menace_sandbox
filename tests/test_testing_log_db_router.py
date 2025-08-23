import menace.bot_testing_bot as btb
import db_router as dr


def test_testing_log_db_uses_router(tmp_path):
    router = dr.DBRouter("tlog", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    db = btb.TestingLogDB(router=router)
    res = btb.TestResult(
        id="1",
        bot="x",
        version="v",
        passed=True,
        error=None,
        timestamp="now",
    )
    db.log(res)
    cur = router.get_connection("results").execute("SELECT id FROM results")
    assert cur.fetchone()[0] == "1"
