import pytest

pytest.importorskip("pandas")

import pandas as pd
import menace.structural_evolution_bot as seb
import menace.data_bot as db
from db_router import init_db_router


def test_predict_and_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    router = init_db_router("evo", str(tmp_path / "e.db"), str(tmp_path / "e.db"))
    bot = seb.StructuralEvolutionBot(mdb, seb.EvolutionDB(router=router))
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
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 200.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    router = init_db_router("evo2", str(tmp_path / "e.db"), str(tmp_path / "e.db"))
    bot = seb.StructuralEvolutionBot(mdb, seb.EvolutionDB(router=router))
    snap = bot.take_snapshot()
    recs = bot.predict_changes(snap)
    if recs[0].severity == "major":
        approved = bot.apply_major_change(recs[0], approve_cb=lambda r: True)
        assert approved
    assert router._access_counts["local"]["evolutions"] >= 1
    router.close()

