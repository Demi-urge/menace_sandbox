from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import json

import menace.ranking_model_scheduler as rms
from vector_service.cognition_layer import CognitionLayer


def test_scheduler_trains_and_reloads(tmp_path, monkeypatch):
    # dummy service recording reload calls
    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    svc = DummyService()

    sched = rms.RankingModelScheduler([svc],
                                      vector_db=tmp_path / "vec.db",
                                      metrics_db=tmp_path / "metrics.db",
                                      model_path=tmp_path / "model.json",
                                      interval=0)

    # stub heavy dependencies
    monkeypatch.setattr(rms.rr, "load_training_data", lambda **kw: object())
    dummy_model = SimpleNamespace(coef_=[[1.0]], intercept_=[0.0], classes_=[0, 1])
    monkeypatch.setattr(
        rms.rr, "train", lambda df: rms.rr.TrainedModel(dummy_model, ["x"])
    )
    monkeypatch.setattr(
        rms.rr, "save_model", lambda tm, p: Path(p).write_text("{}")
    )

    stats_called: list[Path] = []
    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: stats_called.append(Path(m)))

    sched.retrain_and_reload()

    cfg = json.loads((tmp_path / "model.json").read_text())
    current = Path(cfg["current"])
    assert svc.model_path == current
    assert svc.reliability_reloaded


def test_scheduler_retrains_on_roi_feedback(tmp_path, monkeypatch):
    class DummyBus:
        def __init__(self) -> None:
            self.cbs = {}
            self.events = []

        def subscribe(self, topic, callback):
            self.cbs.setdefault(topic, []).append(callback)

        def publish(self, topic, event):
            self.events.append((topic, event))
            for cb in self.cbs.get(topic, []):
                cb(topic, event)

    bus = DummyBus()

    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    svc = DummyService()

    sched = rms.RankingModelScheduler(
        [svc],
        vector_db=tmp_path / "vec.db",
        metrics_db=tmp_path / "metrics.db",
        model_path=tmp_path / "model.json",
        event_bus=bus,
        roi_signal_threshold=1.0,
        win_rate_threshold=0.5,
    )

    monkeypatch.setattr(rms.rr, "load_training_data", lambda **kw: object())
    dummy_model = SimpleNamespace(coef_=[[1.0]], intercept_=[0.0], classes_=[0, 1])
    calls: list[int] = []

    def train(df):
        calls.append(1)
        return rms.rr.TrainedModel(dummy_model, ["x"])

    monkeypatch.setattr(rms.rr, "train", train)
    monkeypatch.setattr(rms.rr, "save_model", lambda tm, p: Path(p).write_text("{}"))
    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: None)
    monkeypatch.setattr(rms, "needs_retrain", lambda db, thr: True)
    monkeypatch.setattr(rms, "build_dataset", lambda **kw: object())

    metrics = rms.VectorMetricsDB(tmp_path / "vec.db")
    layer = CognitionLayer(
        context_builder=SimpleNamespace(refresh_db_weights=lambda *a, **k: None),
        patch_logger=SimpleNamespace(event_bus=bus),
        vector_metrics=metrics,
        event_bus=bus,
    )

    vectors = [("db", "v1", 0.0)]
    layer.update_ranker(
        vectors,
        True,
        roi_deltas={"db": 0.4},
        risk_scores={"db": 0.1},
    )
    assert not calls
    first_events = list(bus.events)
    assert first_events and first_events[-1][1]["risk"] == 0.1

    layer.update_ranker(
        vectors,
        False,
        roi_deltas={"db": 0.7},
        risk_scores={"db": 0.2},
    )
    assert calls
    new_events = [e for e in bus.events[len(first_events):] if e[0] == "retrieval:feedback"]
    assert new_events and new_events[-1][1]["risk"] == 0.2
    cfg = json.loads((tmp_path / "model.json").read_text())
    current = Path(cfg["current"])
    assert svc.model_path == current
    assert svc.reliability_reloaded
    assert current.exists()


def test_record_patch_outcome_feedback_triggers_retrain(tmp_path, monkeypatch):
    class DummyBus:
        def __init__(self) -> None:
            self.cbs = {}
            self.events = []

        def subscribe(self, topic, callback):
            self.cbs.setdefault(topic, []).append(callback)

        def publish(self, topic, event):
            self.events.append((topic, event))
            for cb in self.cbs.get(topic, []):
                cb(topic, event)

    bus = DummyBus()
    calls: list[int] = []
    sched = rms.RankingModelScheduler(
        [],
        vector_db=tmp_path / "vec.db",
        metrics_db=tmp_path / "metrics.db",
        model_path=tmp_path / "model.json",
        event_bus=bus,
        roi_signal_threshold=1.5,
    )
    monkeypatch.setattr(rms, "needs_retrain", lambda db, thr: True)
    sched.retrain_and_reload = lambda: calls.append(1)  # type: ignore

    metrics = rms.VectorMetricsDB(tmp_path / "vec.db")
    patch_logger = SimpleNamespace(event_bus=bus, track_contributors=lambda *a, **k: {})
    layer = CognitionLayer(
        context_builder=SimpleNamespace(refresh_db_weights=lambda *a, **k: None),
        patch_logger=patch_logger,
        vector_metrics=metrics,
        event_bus=bus,
    )

    sid = "s1"
    layer._session_vectors[sid] = [("db", "v1", 0.0)]
    layer._retrieval_meta[sid] = {"db:v1": {"risk_score": 0.3}}

    layer.record_patch_outcome(sid, True, contribution=1.0)

    assert len(calls) == 1
    assert any(
        topic == "retrieval:feedback" and event.get("risk") == 0.3
        for topic, event in bus.events
    )


def test_scheduler_reloads_dependents(tmp_path, monkeypatch):
    class ChildService:
        def __init__(self) -> None:
            self.reliability_reloaded = False
            self.model_path: Path | None = None
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    child = ChildService()

    class ParentService:
        def __init__(self) -> None:
            self.dependent_services = [child]
            self.reliability_reloaded = False
            self.model_path: Path | None = None
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    parent = ParentService()

    sched = rms.RankingModelScheduler([parent],
                                      vector_db=tmp_path / "vec.db",
                                      metrics_db=tmp_path / "metrics.db",
                                      model_path=tmp_path / "model.json",
                                      interval=0)

    monkeypatch.setattr(rms.rr, "load_training_data", lambda **kw: object())
    dummy_model = SimpleNamespace(coef_=[[1.0]], intercept_=[0.0], classes_=[0, 1])
    monkeypatch.setattr(rms.rr, "train", lambda df: rms.rr.TrainedModel(dummy_model, ["x"]))
    monkeypatch.setattr(rms.rr, "save_model", lambda tm, p: Path(p).write_text("{}"))
    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: None)

    sched.retrain_and_reload()

    assert parent.reliability_reloaded
    assert child.reliability_reloaded


def test_scheduler_retrains_on_roi_signal(monkeypatch):
    class Tracker(SimpleNamespace):
        def __init__(self) -> None:
            super().__init__()
            self.origin_db_delta_history = {"db": []}
            self.raroi_history = []

        def origin_db_deltas(self):
            return {
                db: vals[-1]
                for db, vals in self.origin_db_delta_history.items()
                if vals
            }

    tracker = Tracker()
    sched = rms.RankingModelScheduler(
        [],
        roi_tracker=tracker,
        interval=100,
        roi_signal_threshold=0.5,
    )

    calls: list[int] = []

    def retrain() -> None:
        calls.append(len(calls))
        if len(calls) >= 2:
            sched.running = False

    monkeypatch.setattr(rms.time, "sleep", lambda s: None)
    sched.retrain_and_reload = retrain  # type: ignore
    sched.running = True
    t = threading.Thread(target=sched._loop)
    t.start()
    # wait for first call
    while len(calls) == 0:
        pass
    tracker.origin_db_delta_history["db"].append(1.0)
    t.join(timeout=0.1)
    assert len(calls) >= 2


def test_scheduler_retrains_on_win_rate_drop(tmp_path, monkeypatch):
    class DummyBus:
        def __init__(self) -> None:
            self.cbs = {}

        def subscribe(self, topic, callback):
            self.cbs[topic] = callback

        def publish(self, topic, event):
            cb = self.cbs.get(topic)
            if cb:
                cb(topic, event)

    bus = DummyBus()

    # Start with perfect win rate
    vdb = rms.VectorMetricsDB(tmp_path / "vec.db")
    vdb.log_retrieval_feedback("db", win=True, regret=False)
    vdb.conn.close()

    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    svc = DummyService()

    sched = rms.RankingModelScheduler(
        [svc],
        vector_db=tmp_path / "vec.db",
        metrics_db=tmp_path / "metrics.db",
        model_path=tmp_path / "model.json",
        event_bus=bus,
        win_rate_threshold=0.5,
    )

    monkeypatch.setattr(rms.rr, "load_training_data", lambda **kw: object())
    dummy_model = SimpleNamespace(coef_=[[1.0]], intercept_=[0.0], classes_=[0, 1])
    calls: list[int] = []

    def train(df):
        calls.append(1)
        return rms.rr.TrainedModel(dummy_model, ["x"])

    monkeypatch.setattr(rms.rr, "train", train)
    monkeypatch.setattr(rms.rr, "save_model", lambda tm, p: Path(p).write_text("{}"))
    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: None)

    # No retrain while win rate above threshold
    bus.publish("retrieval:feedback", {"db": "db", "win": True, "regret": False})
    assert not calls

    # Record losses to drop win rate below threshold and publish event
    vdb = rms.VectorMetricsDB(tmp_path / "vec.db")
    vdb.log_retrieval_feedback("db", win=False, regret=True)
    vdb.log_retrieval_feedback("db", win=False, regret=True)
    vdb.conn.close()
    bus.publish("retrieval:feedback", {"db": "db", "win": False, "regret": True})

    assert calls
    cfg = json.loads((tmp_path / "model.json").read_text())
    current = Path(cfg["current"])
    assert svc.model_path == current
    assert svc.reliability_reloaded

