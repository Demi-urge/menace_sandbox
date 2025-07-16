import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types, sys
stub = types.ModuleType("jinja2")
stub.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", stub)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
err_mod = types.ModuleType("menace.error_bot")
class DummyErr:
    pass
err_mod.ErrorBot = DummyErr
sys.modules.setdefault("menace.error_bot", err_mod)

from menace.capital_management_bot import SummaryHistoryDB, SummaryRecord

def test_log_and_fetch(tmp_path):
    db = SummaryHistoryDB(tmp_path / "s.db")
    rec = SummaryRecord(run_id="x", capital=100.0, trend=0.1, energy=0.5, message="hi")
    db.log_summary(rec)
    rows = db.fetch()
    assert rows
    assert rows[0][0] == "x"
