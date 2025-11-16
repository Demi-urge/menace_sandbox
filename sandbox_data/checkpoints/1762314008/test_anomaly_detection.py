import sys
import types

import pytest

pytest.importorskip("pandas")
import pandas as pd


class DummyMetricsDB:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, float]] = []

    def log_eval(self, cycle: str, metric: str, value: float) -> None:
        self.records.append((cycle, metric, value))


def _prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name.startswith("menace") or name in {"dynamic_path_router", "sandbox_runner"}:
            sys.modules.pop(name, None)
    stub = types.ModuleType("stub")
    monkeypatch.setitem(sys.modules, "psutil", stub)
    monkeypatch.setitem(sys.modules, "sklearn", stub)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", stub)
    stub.KMeans = object
    si = types.ModuleType("menace.self_improvement")
    bt = types.ModuleType("menace.self_improvement.baseline_tracker")
    bt.BaselineTracker = object
    si.baseline_tracker = bt
    monkeypatch.setitem(sys.modules, "menace.self_improvement", si)
    monkeypatch.setitem(sys.modules, "menace.self_improvement.baseline_tracker", bt)
    monkeypatch.setitem(sys.modules, "sandbox_runner", types.ModuleType("sandbox_runner"))
    monkeypatch.setitem(sys.modules, "sandbox_runner.bootstrap", types.ModuleType("sandbox_runner.bootstrap"))
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", types.ModuleType("sandbox_runner.cli"))
    monkeypatch.setitem(sys.modules, "menace.metrics_dashboard", types.ModuleType("menace.metrics_dashboard"))
    monkeypatch.setitem(
        sys.modules,
        "menace.code_database",
        types.SimpleNamespace(PatchHistoryDB=object),
    )


def test_anomaly_scores_mad(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    from menace import anomaly_detection as ad

    monkeypatch.setattr(ad, "torch", None)
    monkeypatch.setattr(ad, "nn", None)
    monkeypatch.setattr(ad, "KMeans", None)
    monkeypatch.setattr(ad, "np", None)
    mdb = DummyMetricsDB()
    scores = ad.anomaly_scores([1.0, 2.0, 50.0], metrics_db=mdb, field="cpu")
    assert len(scores) == 3
    assert scores[-1] > scores[0]
    assert len(mdb.records) == 3


def test_anomaly_scores_empty() -> None:
    from menace import anomaly_detection as ad

    assert ad.anomaly_scores([]) == []


def test_anomaly_scores_extreme_outlier(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    from menace import anomaly_detection as ad

    monkeypatch.setattr(ad, "torch", None)
    monkeypatch.setattr(ad, "nn", None)
    monkeypatch.setattr(ad, "KMeans", None)
    monkeypatch.setattr(ad, "np", None)
    scores = ad.anomaly_scores([1.0, 2.0, 10 ** 9])
    assert scores[-1] > scores[0] * 1_000_000


def test_anomaly_scores_logs_metrics_advanced(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    from menace import anomaly_detection as ad

    monkeypatch.setattr(ad, "torch", None)
    monkeypatch.setattr(ad, "nn", None)
    monkeypatch.setattr(ad, "_cluster_scores", lambda vals: [42.0 for _ in vals])
    monkeypatch.setattr(ad, "np", object())
    mdb = DummyMetricsDB()
    scores = ad.anomaly_scores([1.0, 2.0], metrics_db=mdb, field="cpu")
    assert scores == [42.0, 42.0]
    assert mdb.records == [("anomaly", "cpu", 42.0), ("anomaly", "cpu", 42.0)]


def test_detect_anomalies_logging(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    import menace.data_bot as db

    mdb = db.MetricsDB(tmp_path / "m.db")
    data = pd.DataFrame({"cpu": [1, 2, 50]})
    idxs = db.DataBot.detect_anomalies(data, "cpu", threshold=1.0, metrics_db=mdb)
    assert idxs == [2]
    rows = mdb.fetch_eval("anomaly")
    assert rows


def test_detect_anomalies_logs_error(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    _prepare(monkeypatch)
    import menace.data_bot as db

    boom_mod = types.ModuleType("menace.anomaly_detection")

    def boom(*a, **k):  # noqa: D401 - simple stub
        raise RuntimeError("fail")

    boom_mod.anomaly_scores = boom
    monkeypatch.setitem(sys.modules, "menace.anomaly_detection", boom_mod)

    data = pd.DataFrame({"cpu": [1, 2, 50]})
    caplog.set_level("ERROR")
    idxs = db.DataBot.detect_anomalies(data, "cpu", threshold=1.0)
    assert idxs == [2]
    assert "anomaly detection failed" in caplog.text

