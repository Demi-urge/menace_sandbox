from menace import mutation_logger as ml
from menace import monitoring_dashboard as md
from menace import evolution_history_db as eh
from menace import error_bot as eb
from menace import data_bot as db


class MockEventBus:
    def __init__(self):
        self.subs = {}
        self.publish_calls = 0

    def subscribe(self, topic, cb):
        self.subs.setdefault(topic, []).append(cb)

    def publish(self, topic, event):
        self.publish_calls += 1
        # fail first publish to exercise retry
        if self.publish_calls == 1:
            raise RuntimeError('fail')
        for cb in self.subs.get(topic, []):
            cb(topic, event)


def test_logger_to_dashboard_propagation(tmp_path, monkeypatch):
    # set up databases
    mdb = db.MetricsDB(tmp_path / 'm.db')
    edb = eh.EvolutionHistoryDB(tmp_path / 'e.db')
    errdb = eb.ErrorDB(tmp_path / 'err.db')
    # patch mutation_logger to use temp db and mock bus
    monkeypatch.setattr(ml, '_history_db', edb)
    bus = MockEventBus()
    ml.set_event_bus(bus)
    dash = md.MonitoringDashboard(mdb, edb, errdb, event_bus=bus)
    ml.log_mutation('root', 'r', 't', 1.0, workflow_id=1, before_metric=0.0, after_metric=1.0)
    data = dash._lineage_updates.get(timeout=1)
    assert data[0]['action'] == 'root'
    assert bus.publish_calls >= 2
