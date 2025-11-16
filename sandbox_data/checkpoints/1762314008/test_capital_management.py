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

import pytest
from menace.capital_management_bot import CapitalManagementBot


def test_profit_trend_description_increase(monkeypatch):
    bot = CapitalManagementBot()
    bot.profit_history.extend([100.0, 110.0])
    assert "increasing" in bot.profit_trend_description()


def test_dynamic_energy_threshold():
    bot = CapitalManagementBot()
    bot.energy_history.extend([0.5, 0.6, 0.4])
    th = bot.dynamic_energy_threshold()
    assert 0.1 <= th <= 1.0


def test_dynamic_energy_threshold_default(monkeypatch):
    monkeypatch.setenv("CM_DEFAULT_ENERGY_THRESHOLD", "0.65")
    bot = CapitalManagementBot()
    bot.energy_history.clear()
    assert bot.dynamic_energy_threshold() == 0.65


def test_binary_weight_env(monkeypatch):
    monkeypatch.setenv("CM_BINARY_WEIGHT_TREND", "0.5")
    monkeypatch.setenv("CM_BINARY_WEIGHT_CAPITAL", "0.25")
    monkeypatch.setenv("CM_BINARY_WEIGHT_VOLATILITY", "0.25")
    bot = CapitalManagementBot()
    w = bot.config.binary_weights
    assert abs(sum(w.values()) - 1.0) < 1e-6
