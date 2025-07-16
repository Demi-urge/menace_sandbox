import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.oversight_bots as ob
import menace.data_bot as db
import menace.capital_management_bot as cb


def test_oversight_monitor(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(mdb)
    capital = cb.CapitalManagementBot(data_bot=data_bot)
    mdb.add(
        db.MetricRecord(
            bot="x", cpu=10.0, memory=10.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=0
        )
    )
    bot = ob.L1OversightBot(data_bot, capital)
    bot.add_subordinate("x")
    df = bot.monitor()
    assert not df.empty
