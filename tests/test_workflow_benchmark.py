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

    def abs(self, arr):
        return [abs(x) for x in arr]

    def std(self, arr):
        return 0.0

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


def test_benchmark_and_training(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    pdb = neu.PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "mem.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore

    def dummy():
        return True

    def dummy_fail():
        return False

    ok = wb.benchmark_workflow(dummy, mdb, pdb, name="test")
    assert ok
    wb.benchmark_workflow(dummy_fail, mdb, pdb, name="test")
    rows = mdb.fetch_eval("test")
    assert rows
    cur = pdb.conn.execute(
        "SELECT success_rate FROM metadata m JOIN pathways p ON p.id=m.pathway_id WHERE p.actions=?",
        ("test",),
    )
    row = cur.fetchone()
    assert row and 0 <= row[0] <= 1

    engine = LearningEngine(pdb, mm)
    assert engine.train()


def test_registered_workflow_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    mdb = db.MetricsDB(tmp_path / "m.db")
    pdb = neu.PathwayDB(tmp_path / "p.db")
    tracker = ROITracker()
    tracker._regression = lambda: (None, [])

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
    assert "latency" in tracker.metrics_history
    rows = mdb.fetch_eval("ok")
    pvalue_metrics = {r[1] for r in rows if r[1].endswith("_pvalue")}
    assert "duration_pvalue" in pvalue_metrics
    assert "cpu_time_pvalue" in pvalue_metrics
    assert "memory_usage_pvalue" in pvalue_metrics
