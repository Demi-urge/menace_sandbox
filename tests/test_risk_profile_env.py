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


def test_aggressive_multiplier(monkeypatch):
    cfg = cmb.CapitalManagementConfig(
        risk_profile="aggressive",
        conserve_threshold=0.5,
        aggressive_threshold=0.8,
    )
    monkeypatch.setenv("CM_RISK_AGGR_MULT", "0.75")
    cfg.apply_risk_profile()
    assert cfg.conserve_threshold == 0.5 * 0.75
    assert cfg.aggressive_threshold == 0.8 * 0.75


def test_conservative_multiplier(monkeypatch):
    cfg = cmb.CapitalManagementConfig(
        risk_profile="conservative",
        conserve_threshold=0.5,
        aggressive_threshold=0.8,
    )
    monkeypatch.setenv("CM_RISK_CONSERV_MULT", "1.5")
    cfg.apply_risk_profile()
    assert cfg.conserve_threshold == 0.5 * 1.5
    assert cfg.aggressive_threshold == 0.8 * 1.5
