
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types, sys
stub = types.ModuleType("jinja2")
stub.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", stub)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
err_mod = types.ModuleType("menace.error_bot")
class DummyErr:  # pragma: no cover - simple stub
    pass
err_mod.ErrorBot = DummyErr
sys.modules.setdefault("menace.error_bot", err_mod)

import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck

import menace.capital_management_bot as cmb
import menace.data_bot as db


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    inflows=st.lists(st.floats(min_value=0, max_value=1e6), max_size=20),
    expenses=st.lists(st.floats(min_value=0, max_value=1e6), max_size=20),
)
def test_profit_non_negative(tmp_path, inflows, expenses):
    bot = cmb.CapitalManagementBot(
        ledger=cmb.CapitalLedger(tmp_path / "l.db"),
        allocation_ledger=cmb.CapitalAllocationLedger(tmp_path / "a.db"),
    )
    for amt in inflows:
        bot.log_inflow(float(amt))
    for amt in expenses:
        bot.log_expense(float(amt))
    assert bot.profit() >= 0


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    actions=st.lists(
        st.tuples(st.sampled_from(["inflow", "expense"]), st.floats(min_value=0, max_value=1e6)),
        max_size=30,
    )
)
def test_profit_trend_non_negative(tmp_path, actions):
    bot = cmb.CapitalManagementBot(
        ledger=cmb.CapitalLedger(tmp_path / "l.db"),
        allocation_ledger=cmb.CapitalAllocationLedger(tmp_path / "a.db"),
    )
    for kind, amt in actions:
        if kind == "inflow":
            bot.log_inflow(float(amt))
        else:
            bot.log_expense(float(amt))
        assert bot.profit_trend() >= 0


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    load=st.floats(min_value=0, max_value=1),
    success=st.floats(min_value=0, max_value=1),
    deploy=st.floats(min_value=0, max_value=1),
    fail=st.floats(min_value=0, max_value=1),
)
def test_energy_score_non_negative(tmp_path, load, success, deploy, fail):
    bot = cmb.CapitalManagementBot(
        ledger=cmb.CapitalLedger(tmp_path / "l.db"),
        allocation_ledger=cmb.CapitalAllocationLedger(tmp_path / "a.db"),
    )
    bot.log_inflow(100.0)
    score = bot.energy_score(
        load=load,
        success_rate=success,
        deploy_eff=deploy,
        failure_rate=fail,
    )
    assert score >= 0


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    records=st.lists(
        st.tuples(st.floats(min_value=0, max_value=1e6), st.floats(min_value=0, max_value=1e6)),
        min_size=1,
        max_size=20,
    )
)
def test_bot_roi_non_negative(tmp_path, records):
    mdb = db.MetricsDB(tmp_path / "m.db")
    for rev, exp in records:
        mdb.add(db.MetricRecord(
            bot="a",
            cpu=0.0,
            memory=0.0,
            response_time=0.0,
            disk_io=0.0,
            net_io=0.0,
            errors=0,
            revenue=float(rev),
            expense=float(exp),
        ))
    data_bot = db.DataBot(mdb)
    bot = cmb.CapitalManagementBot(
        ledger=cmb.CapitalLedger(tmp_path / "l.db"),
        allocation_ledger=cmb.CapitalAllocationLedger(tmp_path / "a.db"),
        data_bot=data_bot,
    )
    assert bot.bot_roi("a") >= 0
