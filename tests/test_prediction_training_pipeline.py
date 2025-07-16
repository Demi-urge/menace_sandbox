import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.prediction_training_pipeline as ptp
import menace.prediction_manager_bot as pmb
import menace.data_bot as db
import menace.capital_management_bot as cmb


class DummyPredict:
    def predict(self, X):
        return [0.0 for _ in X]


def test_training_pipeline(tmp_path):
    metrics_db = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(metrics_db)
    capital = cmb.CapitalManagementBot(data_bot=data_bot)
    manager = pmb.PredictionManager(tmp_path / "reg.json", data_bot=data_bot, capital_bot=capital)
    bid = manager.register_bot(DummyPredict(), {"risk": ["low"]})
    metrics_db.add(db.MetricRecord(bot=bid, cpu=1.0, memory=1.0, response_time=0.1, disk_io=0.0, net_io=0.0, errors=10))
    pipeline = ptp.PredictionTrainingPipeline(manager=manager, data_bot=data_bot, capital_bot=capital, max_generations=1, threshold=0.5)
    res = pipeline.train([bid])
    assert res and res[0].generations >= 1
    assert manager.registry.get(res[0].bot_id)
