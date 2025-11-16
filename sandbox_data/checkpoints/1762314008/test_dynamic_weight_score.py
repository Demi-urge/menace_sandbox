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

def test_dynamic_weighted_score_custom_weights():
    weights = {"capital": 1.0, "roi": 0.0, "volatility": 0.0, "risk": 0.0, "stddev": 0.0}
    score = cmb.dynamic_weighted_energy_score(
        capital=100.0,
        roi=0.5,
        volatility=0.0,
        risk_profile="balanced",
        performance_stddev=0.0,
        norm_factor=100.0,
        weights=weights,
        log_func=lambda x: x,
        clamp=lambda x: x,
    )
    assert score == 1.0


def test_run_id_present():
    bot = cmb.CapitalManagementBot()
    assert bot.run_id

