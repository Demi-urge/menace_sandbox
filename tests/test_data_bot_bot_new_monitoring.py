import types
from menace import data_bot as db
from menace.roi_thresholds import ROIThresholds

class Bus:
    def __init__(self):
        self.subs = {}
    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)
    def publish(self, topic, payload):
        for fn in self.subs.get(topic, []):
            fn(topic, payload)

def test_bot_new_persists_monitoring(tmp_path, monkeypatch):
    db_path = tmp_path / 'metrics.db'
    metrics = db.MetricsDB(db_path)
    bus = Bus()

    calls = []
    def fake_reload(self, bot=None):
        calls.append(bot)
        return ROIThresholds(roi_drop=-0.1, error_threshold=1.0)
    monkeypatch.setattr(db.DataBot, 'reload_thresholds', fake_reload, raising=False)

    bot = db.DataBot(db=metrics, event_bus=bus, start_server=False)
    bus.publish('bot:new', {'name': 'alpha'})

    assert 'alpha' in metrics.monitored_bots()
    assert calls and calls[-1] == 'alpha'
    assert 'alpha' in bot._baseline

    # simulate restart using same db
    bus2 = Bus()
    bot2 = db.DataBot(db=db.MetricsDB(db_path), event_bus=bus2, start_server=False)
    assert 'alpha' in bot2._baseline
