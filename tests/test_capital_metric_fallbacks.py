import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types, sys
stub = types.ModuleType("jinja2")
stub.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", stub)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
err_mod = types.ModuleType("menace.error_bot")
class DummyErr:
    def __init__(self):
        self.flagged = False
    def flag_module(self, name):
        self.flagged = True
err_mod.ErrorBot = DummyErr
sys.modules.setdefault("menace.error_bot", err_mod)

import menace.capital_management_bot as cmb


def test_fetch_metric_fallback(tmp_path, monkeypatch):
    captured = {}
    def fake_alert(msg: str, webhook: str) -> bool:
        captured['msg'] = msg
        captured['webhook'] = webhook
        return True
    monkeypatch.setattr("menace.capital_management_bot.send_discord_alert", fake_alert)
    err = DummyErr()
    val = cmb.fetch_metric_from_db(
        "missing",
        tmp_path / "no.db",
        default=1.23,
        error_bot=err,
        webhook_url="http://hook"
    )
    assert val == 1.23
    assert err.flagged
    assert "missing" in captured['msg']


def test_get_metrics_fallback(tmp_path, monkeypatch):
    captured = {}
    def fake_alert(msg: str, webhook: str) -> bool:
        captured['msg'] = msg
        captured['webhook'] = webhook
        return True
    monkeypatch.setattr("menace.capital_management_bot.send_discord_alert", fake_alert)
    err = DummyErr()
    cfg = cmb.CapitalManagementConfig()
    cfg.metrics_db_path = str(tmp_path / "missing.db")
    cfg.metric_fallbacks = {
        "capital": 0.0,
        "profit_trend": 0.0,
        "load": 0.5,
        "success_rate": 0.8,
        "deploy_efficiency": 0.7,
        "failure_rate": 0.2,
    }
    bot = cmb.CapitalManagementBot(
        ledger=cmb.CapitalLedger(tmp_path / "l.db"),
        allocation_ledger=cmb.CapitalAllocationLedger(tmp_path / "a.db"),
        error_bot=err,
        webhook_url="http://hook",
        config=cfg,
    )
    bot.log_inflow(100.0)
    metrics = bot.get_metrics()
    assert metrics["load"] == 0.5
    assert err.flagged
    assert captured['webhook'] == "http://hook"


def test_dynamic_score_range():
    score = cmb.dynamic_weighted_energy_score(
        capital=100.0,
        roi=0.2,
        volatility=0.1,
        risk_profile="balanced",
        performance_stddev=0.05,
    )
    assert 0.0 <= score <= 1.0
