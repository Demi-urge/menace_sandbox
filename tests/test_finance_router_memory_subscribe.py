import os
import logging
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
capital_stub = types.ModuleType("menace.capital_management_bot")
capital_stub.CapitalManagementBot = object
sys.modules.setdefault("menace.capital_management_bot", capital_stub)
import menace.finance_router_bot as frb  # noqa: E402


def test_finance_router_memory_subscription_error(tmp_path, caplog):
    class BadMem:
        def subscribe(self, *a, **k):
            raise RuntimeError("fail")

    caplog.set_level(logging.ERROR)
    frb.FinanceRouterBot(payout_log_path=tmp_path / "p.json", memory_mgr=BadMem())
    assert "memory subscription failed" in caplog.text
