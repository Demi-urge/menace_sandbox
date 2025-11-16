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

import menace.capital_management_bot as cmb


def test_threshold_step_env(monkeypatch):
    monkeypatch.setenv("CM_THRESHOLD_STEP", "0.25")
    bot = cmb.CapitalManagementBot()
    bot.energy_history.extend([0.6, 0.6])
    prev_cons = bot.config.conserve_threshold
    prev_aggr = bot.config.aggressive_threshold
    bot._adapt_thresholds()
    assert bot.config.threshold_step == 0.2
    expected_cons = max(0.1, min(0.9, prev_cons + 0.1 * 0.2))
    expected_aggr = max(
        expected_cons + 0.2,
        min(0.95, prev_aggr + 0.1 * 0.2),
    )
    assert bot.config.conserve_threshold == expected_cons
    assert bot.config.aggressive_threshold == expected_aggr
