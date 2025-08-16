from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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

