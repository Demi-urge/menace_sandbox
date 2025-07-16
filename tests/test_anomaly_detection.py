import types
import sys
import types
import pytest

pytest.importorskip("pandas")
import pandas as pd


def _prepare(monkeypatch):
    # stub heavy optional deps before importing menace
    stub = types.ModuleType("stub")
    monkeypatch.setitem(sys.modules, "psutil", stub)
    monkeypatch.setitem(sys.modules, "sklearn", stub)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", stub)
    stub.KMeans = object


def test_anomaly_scores_fallback(monkeypatch):
    _prepare(monkeypatch)
    from menace import anomaly_detection as ad

    monkeypatch.setattr(ad, "torch", None)
    monkeypatch.setattr(ad, "nn", None)
    monkeypatch.setattr(ad, "KMeans", None)
    monkeypatch.setattr(ad, "np", None)
    scores = ad.anomaly_scores([1.0, 2.0, 50.0])
    assert len(scores) == 3
    assert scores[-1] > scores[0]
    # cleanup imported modules
    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)


def test_detect_anomalies_logging(tmp_path, monkeypatch):
    _prepare(monkeypatch)
    import menace.data_bot as db

    mdb = db.MetricsDB(tmp_path / "m.db")
    data = pd.DataFrame({"cpu": [1, 2, 50]})
    idxs = db.DataBot.detect_anomalies(data, "cpu", threshold=1.0, metrics_db=mdb)
    assert idxs == [2]
    rows = mdb.fetch_eval("anomaly")
    assert rows
    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)


def test_detect_anomalies_logs_error(monkeypatch, caplog):
    _prepare(monkeypatch)
    import menace.data_bot as db

    boom_mod = types.ModuleType("menace.anomaly_detection")

    def boom(*a, **k):
        raise RuntimeError("fail")

    boom_mod.anomaly_scores = boom
    monkeypatch.setitem(sys.modules, "menace.anomaly_detection", boom_mod)

    data = pd.DataFrame({"cpu": [1, 2, 50]})
    caplog.set_level("ERROR")
    idxs = db.DataBot.detect_anomalies(data, "cpu", threshold=1.0)
    assert idxs == [2]
    assert "anomaly detection failed" in caplog.text

    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)
