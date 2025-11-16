import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)

from menace.trend_predictor import TrendPredictor
from menace.data_bot import MetricsDB, MetricRecord
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent


def test_predict_future_metrics(tmp_path):
    mdb = MetricsDB(tmp_path / "m.db")
    hdb = EvolutionHistoryDB(tmp_path / "e.db")
    mdb.add(MetricRecord(bot="x", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=1, revenue=1.0, expense=0.5))
    mdb.add(MetricRecord(bot="x", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=2, revenue=1.2, expense=0.6))
    hdb.add(EvolutionEvent(action="a", before_metric=0.0, after_metric=0.0, roi=0.5, efficiency=0.0, bottleneck=1.0))
    predictor = TrendPredictor(history_db=hdb, metrics_db=mdb)
    predictor.train()
    pred = predictor.predict_future_metrics(1)
    assert isinstance(pred.roi, float)
    assert isinstance(pred.errors, float)
