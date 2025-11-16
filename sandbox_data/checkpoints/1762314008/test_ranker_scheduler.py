from __future__ import annotations

import types, sys

# Stub heavy optional dependencies so importing the package remains lightweight
sys.modules.setdefault("sklearn", types.SimpleNamespace())
sys.modules.setdefault("sklearn.linear_model", types.SimpleNamespace(Ridge=object))
sys.modules.setdefault("scipy", types.SimpleNamespace())
sys.modules.setdefault("scipy.stats", types.SimpleNamespace(ks_2samp=lambda *a, **k: (0, 0)))
sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBRegressor=object))

import menace.analytics.ranker_scheduler as rs


class DummyBus:
    def __init__(self) -> None:
        self.cbs = {}

    def subscribe(self, topic, callback):
        self.cbs.setdefault(topic, []).append(callback)

    def publish(self, topic, event):
        for cb in self.cbs.get(topic, []):
            cb(topic, event)


class DummyService:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def reload_ranker_model(self):
        self.calls.append(1)


def test_feedback_event_triggers_retrain(monkeypatch):
    retrain_calls: list[tuple] = []
    bus = DummyBus()
    svc = DummyService()
    sched = rs.RankerScheduler([svc], event_bus=bus, interval=1000, roi_threshold=0.5)
    monkeypatch.setattr(rs.rvr, "retrain", lambda *a, **k: retrain_calls.append(a))
    bus.publish(rs.ROI_TOPIC, {"roi": 1.0, "db": "x"})
    assert retrain_calls == [(["x"],)]
    assert svc.calls == [1]


def test_risk_event_triggers_retrain(monkeypatch):
    retrain_calls: list[tuple] = []
    bus = DummyBus()
    svc = DummyService()
    sched = rs.RankerScheduler([svc], event_bus=bus, interval=1000, risk_threshold=0.5)
    monkeypatch.setattr(rs.rvr, "retrain", lambda *a, **k: retrain_calls.append(a))
    bus.publish(rs.RISK_TOPIC, {"risk": 1.0, "db": "x"})
    assert retrain_calls == [(["x"],)]
    assert svc.calls == [1]


def test_scheduler_runs_interval(monkeypatch):
    calls: list[int] = []
    svc = DummyService()
    sched = rs.RankerScheduler([svc], interval=0)
    monkeypatch.setattr(rs.rvr, "retrain", lambda *a, **k: calls.append(1))
    sched.start()
    sched._thread.join(timeout=0.1)
    sched.stop()
    assert calls == [1]
    assert svc.calls == [1]
