import os
import types
import sys

os.environ["MENACE_LIGHT_IMPORTS"] = "1"
class DummyNP(types.ModuleType):
    def array(self, data, dtype=float):
        return list(data)

    def arange(self, n):
        return list(range(n))

    def percentile(self, data, q):
        data = sorted(data)
        k = (len(data) - 1) * q / 100
        f = int(k)
        return float(data[f])

    def median(self, data):
        data = sorted(data)
        mid = len(data) // 2
        if len(data) % 2:
            return float(data[mid])
        return float((data[mid - 1] + data[mid]) / 2)

    def abs(self, arr):
        return [abs(x) for x in arr]

    def std(self, arr):
        return 0.0

    def min(self, arr):
        return float(min(arr)) if arr else 0.0

    def max(self, arr):
        return float(max(arr)) if arr else 0.0

np_stub = DummyNP("numpy")
sys.modules["numpy"] = np_stub
sys.modules.setdefault("numpy", np_stub)
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
lin_mod = types.ModuleType("linear_model")
class DummyLR:
    def fit(self, *a, **k):
        pass

    def predict(self, x):
        return [0.0] * len(x)

lin_mod.LinearRegression = DummyLR
sys.modules.setdefault("sklearn.linear_model", lin_mod)
prep_mod = types.ModuleType("preprocessing")
class DummyPF:
    def __init__(self, *a, **k):
        pass

prep_mod.PolynomialFeatures = DummyPF
sys.modules.setdefault("sklearn.preprocessing", prep_mod)

import menace.workflow_benchmark as wb
import menace.data_bot as db
import menace.neuroplasticity as neu
from menace.learning_engine import LearningEngine
from menace.menace_memory_manager import MenaceMemoryManager
from menace.roi_tracker import ROITracker
from types import SimpleNamespace
import pytest


def test_benchmark_and_training(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    pdb = neu.PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore

    def dummy():
        return True

    def dummy_fail():
        return False

    def dummy_crash():  # type: ignore[return-type]
        raise RuntimeError("boom")

    ok = wb.benchmark_workflow(dummy, mdb, pdb, name="test")
    assert ok
    wb.benchmark_workflow(dummy_fail, mdb, pdb, name="test")
    assert not wb.benchmark_workflow(dummy_crash, mdb, pdb, name="crash")
    rows = mdb.fetch_eval("test")
    assert rows
    crash_rows = [r for r in mdb.fetch_eval("crash") if r[1] == "crash"]
    assert crash_rows and crash_rows[-1][2] == 1.0
    cur = pdb.conn.execute(
        "SELECT success_rate FROM metadata m JOIN pathways p ON p.id=m.pathway_id WHERE p.actions=?",
        ("test",),
    )
    row = cur.fetchone()
    assert row and 0 <= row[0] <= 1

    engine = LearningEngine(pdb, mm)
    assert engine.train()


@pytest.mark.skip(reason="requires full runtime environment")
def test_registered_workflow_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    mdb = db.MetricsDB(tmp_path / "m.db")
    pdb = neu.PathwayDB(tmp_path / "p.db")
    tracker = ROITracker(
        telemetry_backend=SimpleNamespace(log_prediction=lambda *a, **k: None)
    )
    tracker._regression = lambda: (None, [])

    import menace.composite_workflow_scorer as cws

    class _DummyResultsDB:
        def __init__(self, *a, **k):
            pass

        def log_result(self, *a, **k):
            pass

        def log_module_attribution(self, *a, **k):
            pass

    monkeypatch.setattr(cws, "ROIResultsDB", _DummyResultsDB)

    def ok():
        return True

    def fail():
        return False

    wb.benchmark_registered_workflows(
        {"ok": ok, "fail": fail},
        [{"env": "prod"}, {}],
        mdb,
        pdb,
        tracker,
    )
    wb.benchmark_registered_workflows(
        {"ok": ok, "fail": fail},
        [{"env": "prod"}, {}],
        mdb,
        pdb,
        tracker,
    )

    assert mdb.fetch_eval("ok")
    assert "duration" in tracker.metrics_history
    assert tracker.metrics_history["success"]
    assert "cpu_time" in tracker.metrics_history
    assert "cpu_percent" in tracker.metrics_history
    assert "memory_delta" in tracker.metrics_history
    assert "memory_usage" in tracker.metrics_history
    assert "memory_peak" in tracker.metrics_history
    assert "latency" in tracker.metrics_history
    assert "latency_median" in tracker.metrics_history
    assert "latency_min" in tracker.metrics_history
    assert "latency_max" in tracker.metrics_history
    assert "cpu_user_time" in tracker.metrics_history
    assert "cpu_system_time" in tracker.metrics_history
    rows = mdb.fetch_eval("ok")
    pvalue_metrics = {r[1] for r in rows if r[1].endswith("_pvalue")}
    assert "duration_pvalue" in pvalue_metrics
    assert "cpu_time_pvalue" in pvalue_metrics
    assert "cpu_user_time_pvalue" in pvalue_metrics
    assert "cpu_system_time_pvalue" in pvalue_metrics
    assert "memory_usage_pvalue" in pvalue_metrics
    assert "memory_peak_pvalue" in pvalue_metrics
    assert "latency_median_pvalue" in pvalue_metrics
    assert "latency_min_pvalue" in pvalue_metrics
    assert "latency_max_pvalue" in pvalue_metrics
