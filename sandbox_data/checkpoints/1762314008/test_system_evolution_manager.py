import pytest
pytest.importorskip("pandas")
import menace.system_evolution_manager as sem
import menace.data_bot as db


def test_run_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 1.0, 1.0, 0.1, 0.1, 0.1, 0)
    mdb.add(rec)
    manager = sem.SystemEvolutionManager(["bot"], metrics_db=mdb)
    manager.struct_bot.metrics_db = mdb
    res = manager.run_cycle()
    assert "bot" in res.ga_results
    assert list(res.predictions)
    rows = mdb.fetch_eval("evolution_cycle")
    assert rows
