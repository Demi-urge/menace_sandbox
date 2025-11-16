import numpy as np
import pandas as pd
import menace.roi_tracker as rt
from menace.analytics.adaptive_roi_dataset import build_dataset
from menace.capital_management_bot import ROIEventDB, ROIEvent
from menace.data_bot import MetricsDB, MetricRecord
from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
import db_router


def test_dataset_dataframe_and_csv(tmp_path):
    tracker = rt.ROITracker()

    t0 = "2024-01-01T00:00:00"
    t1 = "2024-01-01T01:00:00"
    t2 = "2024-01-01T02:00:00"

    router = db_router.DBRouter(
        "analytics", str(tmp_path / "analytics.sqlite"), str(tmp_path / "analytics.sqlite")
    )
    roi_db = ROIEventDB(router=router)
    tracker.update(1.0, 1.1, ["mod"])
    roi_db.add(ROIEvent("mod", 1.0, 1.1, ts=t1))
    tracker.update(1.1, 0.9, ["mod"])
    roi_db.add(ROIEvent("mod", 1.1, 0.9, ts=t2))

    mdb = MetricsDB(router=router)
    mdb.add(MetricRecord("mod", 0, 0, 0, 0, 0, 0, profitability=10.0, ts=t0))
    mdb.add(MetricRecord("mod", 0, 0, 0, 0, 0, 0, profitability=12.0, ts=t1))
    mdb.add(MetricRecord("mod", 0, 0, 0, 0, 0, 0, profitability=11.0, ts=t2))

    eva = EvaluationHistoryDB(router=router)
    eva.add(EvaluationRecord(engine="mod", cv_score=0.8, ts=t1, passed=True))
    eva.add(EvaluationRecord(engine="mod", cv_score=0.4, ts=t2, passed=True))

    df = build_dataset(router=router)

    assert list(df.columns) == ["module", "ts", "roi_delta", "performance_delta", "gpt_score"]
    assert len(df) == 2

    roi_mean = np.mean(tracker.roi_history)
    roi_std = np.std(tracker.roi_history, ddof=1)
    expected_roi = [(v - roi_mean) / roi_std for v in tracker.roi_history]
    assert np.allclose(df["roi_delta"], expected_roi)

    perf_raw = [2.0, -1.0]
    perf_mean = np.mean(perf_raw)
    perf_std = np.std(perf_raw, ddof=1)
    expected_perf = [(v - perf_mean) / perf_std for v in perf_raw]
    assert np.allclose(df["performance_delta"], expected_perf)

    gpt_raw = [0.8, 0.4]
    gpt_mean = np.mean(gpt_raw)
    gpt_std = np.std(gpt_raw, ddof=1)
    expected_gpt = [(v - gpt_mean) / gpt_std for v in gpt_raw]
    assert np.allclose(df["gpt_score"], expected_gpt)

    assert abs(df["roi_delta"].mean()) < 1e-9
    assert abs(df["performance_delta"].mean()) < 1e-9
    assert abs(df["gpt_score"].mean()) < 1e-9

    csv_path = tmp_path / "out.csv"
    df.to_csv(csv_path, index=False)
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "module,ts,roi_delta,performance_delta,gpt_score"
    assert len(lines) == 3
