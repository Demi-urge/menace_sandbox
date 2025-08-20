from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading

import menace.ranking_model_scheduler as rms


def test_scheduler_trains_and_reloads(tmp_path, monkeypatch):
    # dummy service recording reload calls
    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False

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

    assert svc.model_path == tmp_path / "model.json"
    assert svc.reliability_reloaded
    assert stats_called == [tmp_path / "metrics.db"]
    assert (tmp_path / "model.json").exists()


def test_scheduler_reloads_dependents(tmp_path, monkeypatch):
    class ChildService:
        def __init__(self) -> None:
            self.reliability_reloaded = False
            self.model_path: Path | None = None

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
    tracker = SimpleNamespace(origin_db_deltas={"db": []}, raroi_history=[])
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
    tracker.origin_db_deltas["db"].append(1.0)
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
    assert svc.model_path == tmp_path / "model.json"
    assert svc.reliability_reloaded

