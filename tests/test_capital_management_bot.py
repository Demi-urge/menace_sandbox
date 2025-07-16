import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.capital_management_bot as cmb


def test_ledger_and_score(tmp_path):
    ledger = cmb.CapitalLedger(tmp_path / "c.db")
    alloc = cmb.CapitalAllocationLedger(tmp_path / "a.db")
    bot = cmb.CapitalManagementBot(ledger, alloc)
    bot.log_inflow(500.0, "sales")
    bot.log_expense(100.0, "costs")
    score = bot.energy_score(load=0.2, success_rate=0.9, deploy_eff=0.8, failure_rate=0.1)
    bot.evaluate(score, prediction="test")
    rows = ledger.fetch()
    alloc_rows = alloc.fetch()
    assert rows and alloc_rows
    assert 0.0 <= score <= 1.0
    assert alloc_rows[0][0] in {"conserve", "invest", "aggressive"}
