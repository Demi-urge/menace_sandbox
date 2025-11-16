import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.prediction_manager_bot as pmb
import menace.data_bot as db


class Dummy:
    def __init__(self, name, profile):
        self.name = name
        self.prediction_profile = profile


def test_assignment_and_evolution(tmp_path):
    manager = pmb.PredictionManager(tmp_path / "reg.json")
    # register a simple bot
    bot_id = manager.register_bot(object(), {"risk": ["low"], "scope": ["niche"]})
    target = Dummy("t1", {"risk": ["low"], "scope": ["niche"]})
    assigned = manager.assign_prediction_bots(target)
    assert bot_id in assigned

    # request with unmatched profile triggers evolution
    target2 = Dummy("t2", {"risk": ["high"], "scope": ["broad"]})
    assigned2 = manager.assign_prediction_bots(target2)
    assert assigned2  # GA should create a bot
    assert assigned2 != assigned


def test_monitor_and_retire(tmp_path):
    metrics_db = db.MetricsDB(tmp_path / "m.db")
    manager = pmb.PredictionManager(tmp_path / "reg.json")
    bid = manager.register_bot(object(), {"risk": ["mid"]})
    # add poor metrics
    metrics_db.add(db.MetricRecord(bot=bid, cpu=1.0, memory=1.0, response_time=10.0, disk_io=0.0, net_io=0.0, errors=10))
    manager.monitor_bot_performance(metrics_db, threshold=0.05)
    assert bid not in manager.registry


def test_monitor_no_pandas(tmp_path, monkeypatch):
    metrics_db = db.MetricsDB(tmp_path / "m2.db")
    manager = pmb.PredictionManager(tmp_path / "reg.json")
    bid = manager.register_bot(object(), {"risk": ["low"]})
    metrics_db.add(
        db.MetricRecord(
            bot=bid,
            cpu=1.0,
            memory=1.0,
            response_time=1.0,
            disk_io=0.0,
            net_io=0.0,
            errors=10,
        )
    )
    monkeypatch.setattr(
        metrics_db,
        "fetch",
        lambda limit=100: [
            {
                "bot": bid,
                "cpu": 1.0,
                "memory": 1.0,
                "response_time": 1.0,
                "disk_io": 0.0,
                "net_io": 0.0,
                "errors": 10,
            }
        ],
    )
    orig_pd = pmb.pd
    monkeypatch.setattr(pmb, "pd", None)
    manager.monitor_bot_performance(metrics_db, threshold=0.05)
    monkeypatch.setattr(pmb, "pd", orig_pd)
    assert bid not in manager.registry
