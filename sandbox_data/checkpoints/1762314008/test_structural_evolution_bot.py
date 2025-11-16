import pytest

pytest.importorskip("pandas")

import pandas as pd
import menace.data_bot as real_db
from db_router import init_db_router
import types
import sys


def _load_bot(monkeypatch):
    stub_cbi = types.SimpleNamespace(self_coding_managed=lambda **_: (lambda cls: cls))
    monkeypatch.setitem(sys.modules, "coding_bot_interface", stub_cbi)
    monkeypatch.setitem(sys.modules, "menace.coding_bot_interface", stub_cbi)
    class DummyManager:
        def __init__(self, *a, **k):
            pass
    stub_mgr_mod = types.SimpleNamespace(SelfCodingManager=DummyManager)
    monkeypatch.setitem(sys.modules, "self_coding_manager", stub_mgr_mod)
    monkeypatch.setitem(sys.modules, "menace.self_coding_manager", stub_mgr_mod)
    class StubDataBot:
        def __init__(self, *a, **k):
            pass
        @staticmethod
        def complexity_score(df):
            return real_db.DataBot.complexity_score(df)
        def average_errors(self, *a, **k):
            return 0.0
    stub_db_mod = types.SimpleNamespace(
        MetricsDB=real_db.MetricsDB, MetricRecord=real_db.MetricRecord, DataBot=StubDataBot
    )
    monkeypatch.setitem(sys.modules, "data_bot", stub_db_mod)
    monkeypatch.setitem(sys.modules, "menace.data_bot", stub_db_mod)
    import menace.structural_evolution_bot as seb
    return seb


def test_predict_and_apply(tmp_path, monkeypatch):
    seb = _load_bot(monkeypatch)
    mdb = real_db.MetricsDB(tmp_path / "m.db")
    rec = real_db.MetricRecord("bot", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    class T:
        roi_drop = 0.0
        error_threshold = 500.0
        test_failure_threshold = 0.0
    monkeypatch.setattr(
        seb.manager,
        "threshold_service",
        types.SimpleNamespace(get=lambda name=None: T()),
        raising=False,
    )
    router = init_db_router("evo", str(tmp_path / "e.db"), str(tmp_path / "e.db"))
    bot = seb.StructuralEvolutionBot(metrics_db=mdb, db=seb.EvolutionDB(router=router))
    snap = bot.take_snapshot()
    preds = bot.predict_changes(snap)
    assert preds
    applied = bot.apply_minor_changes()
    if preds[0].severity == "minor":
        assert applied == [preds[0].change]
    else:
        assert applied == []
    assert router._access_counts["local"]["evolutions"] >= 1
    router.close()


def test_major_change(tmp_path, monkeypatch):
    seb = _load_bot(monkeypatch)
    mdb = real_db.MetricsDB(tmp_path / "m.db")
    rec = real_db.MetricRecord("bot", 200.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    class T:
        roi_drop = 0.0
        error_threshold = 10.0
        test_failure_threshold = 0.0
    monkeypatch.setattr(
        seb.manager,
        "threshold_service",
        types.SimpleNamespace(get=lambda name=None: T()),
        raising=False,
    )
    router = init_db_router("evo2", str(tmp_path / "e.db"), str(tmp_path / "e.db"))
    bot = seb.StructuralEvolutionBot(metrics_db=mdb, db=seb.EvolutionDB(router=router))
    snap = bot.take_snapshot()
    recs = bot.predict_changes(snap)
    if recs[0].severity == "major":
        approved = bot.apply_major_change(recs[0], approve_cb=lambda r: True)
        assert approved
    assert router._access_counts["local"]["evolutions"] >= 1
    router.close()
