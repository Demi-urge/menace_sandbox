import menace.capital_management_bot as cmb
from menace.db_router import init_db_router
from unittest.mock import patch


def test_fetch_metric_uses_router(tmp_path):
    router = init_db_router("cm", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    conn = router.get_connection("metrics")
    conn.execute(
        "CREATE TABLE metrics(name TEXT, value REAL, ts TEXT, source_menace_id TEXT)"
    )
    conn.execute(
        "INSERT INTO metrics VALUES('m', 1.0, '2024', ?)",
        (router.menace_id,),
    )
    conn.commit()
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        val = cmb.fetch_metric_from_db("m", router=router)
        assert val == 1.0
        gc.assert_called_with("metrics")


def test_capital_ledger_uses_router(tmp_path):
    router = init_db_router("cm2", str(tmp_path / "loc.db"), str(tmp_path / "sha.db"))
    with patch.object(router, "get_connection", wraps=router.get_connection) as gc:
        ledger = cmb.CapitalLedger(router=router)
        ledger.log(cmb.LedgerEntry("inflow", 5.0, "sale"))
        gc.assert_called_with("ledger")
