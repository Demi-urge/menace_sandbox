import pytest

pytest.importorskip("pandas")

import pandas as pd
import menace.structural_evolution_bot as seb
import menace.data_bot as db


def test_predict_and_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    bot = seb.StructuralEvolutionBot(mdb, seb.EvolutionDB(tmp_path / "e.db"))
    snap = bot.take_snapshot()
    preds = bot.predict_changes(snap)
    assert preds
    applied = bot.apply_minor_changes()
    if preds[0].severity == "minor":
        assert applied == [preds[0].change]
    else:
        assert applied == []


def test_major_change(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 200.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    bot = seb.StructuralEvolutionBot(mdb, seb.EvolutionDB(tmp_path / "e.db"))
    snap = bot.take_snapshot()
    recs = bot.predict_changes(snap)
    if recs[0].severity == "major":
        approved = bot.apply_major_change(recs[0], approve_cb=lambda r: True)
        assert approved

